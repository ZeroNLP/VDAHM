# generation.py
import copy
import torch
import transformers
from transformers.generation.utils import *
from transformers.generation.utils import _split_model_inputs
from transformers.generation.configuration_utils import *

from vdahm import replace_attn as replace_attn_vdahm

GenerationMode.VDAHM_GENERATION = "vdahm_generation"

@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    use_model_defaults: Optional[bool] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config, use_model_defaults, **kwargs
    )
    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
            raise ValueError("assisted generate is not supported with Static cache classes`")
        if self._is_stateful:
            # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
            # which is not possible with stateful models (they can't reset to a previous subset of generated text)
            raise ValueError(
                f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
            )

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            target_tokenizer=tokenizer,
            assistant_tokenizer=assistant_tokenizer,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result = self._assisted_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.VDAHM_GENERATION:
        replace_attn_vdahm(self,
                        target_layers=generation_config.vdahm_layers, 
                        aug_ratio=generation_config.vdahm_aug_ratio,
                        filter=generation_config.vdahm_filter)
        result = self._vdahm_decoding(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
        replace_attn_vdahm(self)

    elif generation_mode == GenerationMode.DOLA_GENERATION:
        if self._is_stateful:
            # DoLa decoding was not designed for stateful models, and would require some changes
            raise ValueError(
                f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )
        result = self._dola_decoding(
            input_ids,
            dola_layers=generation_config.dola_layers,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")
        if self._is_stateful:
            # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
            raise ValueError(
                f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._contrastive_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 12. run beam sample
        result = self._beam_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is True
        and hasattr(result, "past_key_values")
        and getattr(result.past_key_values, "to_legacy_cache") is not None
    ):
        result.past_key_values = result.past_key_values.to_legacy_cache()
    return result

def replace_generation():
    transformers.generation.utils.GenerationMixin.generate = generate
    transformers.generation.utils.GenerationMixin._vdahm_decoding = _vdahm_decoding
    transformers.generation.configuration_utils.GenerationConfig.get_generation_mode = get_generation_mode
    transformers.generation.configuration_utils.GenerationConfig.__init__ = generation_config_init



def generation_config_init(self, **kwargs):
    # Parameters that control the length of the output
    self.max_length = kwargs.pop("max_length", 20)
    self.max_new_tokens = kwargs.pop("max_new_tokens", None)
    self.min_length = kwargs.pop("min_length", 0)
    self.min_new_tokens = kwargs.pop("min_new_tokens", None)
    self.early_stopping = kwargs.pop("early_stopping", False)
    self.max_time = kwargs.pop("max_time", None)
    self.stop_strings = kwargs.pop("stop_strings", None)

    # Parameters that control the generation strategy used
    self.do_sample = kwargs.pop("do_sample", False)
    self.num_beams = kwargs.pop("num_beams", 1)
    self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
    self.penalty_alpha = kwargs.pop("penalty_alpha", None)
    self.dola_layers = kwargs.pop("dola_layers", None)

    # VDAHM generation
    self.vdahm_layers = kwargs.pop("vdahm_layers", None)
    self.vdahm_aug_ratio = kwargs.pop("vdahm_aug_ratio", 1.)
    self.vdahm_filter = kwargs.pop("vdahm_filter", False)
    self.vdahm_image_ranges = kwargs.pop("vdahm_image_ranges", None)
    
    # Parameters that control the cache
    self.use_cache = kwargs.pop("use_cache", True)
    self.cache_implementation = kwargs.pop("cache_implementation", None)
    self.cache_config = kwargs.pop("cache_config", None)
    if self.cache_implementation is not None and self.cache_implementation in CACHE_CONFIG_MAPPING:
        cache_config_class = CACHE_CONFIG_MAPPING[self.cache_implementation]
        if isinstance(self.cache_config, dict):
            self.cache_config = cache_config_class.from_dict(self.cache_config)
    self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)
    self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

    # Parameters for manipulation of the model output logits
    self.temperature = kwargs.pop("temperature", 1.0)
    self.top_k = kwargs.pop("top_k", 50)
    self.top_p = kwargs.pop("top_p", 1.0)
    self.min_p = kwargs.pop("min_p", None)
    self.typical_p = kwargs.pop("typical_p", 1.0)
    self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
    self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
    self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
    self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
    self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
    self.length_penalty = kwargs.pop("length_penalty", 1.0)
    self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
    self.bad_words_ids = kwargs.pop("bad_words_ids", None)
    self.force_words_ids = kwargs.pop("force_words_ids", None)
    self.renormalize_logits = kwargs.pop("renormalize_logits", False)
    self.constraints = kwargs.pop("constraints", None)
    self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
    self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
    self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
    self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
    self.suppress_tokens = kwargs.pop("suppress_tokens", None)
    self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
    self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
    self.sequence_bias = kwargs.pop("sequence_bias", None)
    self.token_healing = kwargs.pop("token_healing", False)
    self.guidance_scale = kwargs.pop("guidance_scale", None)
    self.low_memory = kwargs.pop("low_memory", None)
    watermarking_config = kwargs.pop("watermarking_config", None)
    if watermarking_config is None:
        self.watermarking_config = None
    elif isinstance(watermarking_config, BaseWatermarkingConfig):
        self.watermarking_config = watermarking_config
    else:
        self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

    # Parameters that define the output variables of `generate`
    self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
    self.output_attentions = kwargs.pop("output_attentions", False)
    self.output_hidden_states = kwargs.pop("output_hidden_states", False)
    self.output_scores = kwargs.pop("output_scores", False)
    self.output_logits = kwargs.pop("output_logits", None)
    self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

    # Special tokens that can be used at generation time
    self.pad_token_id = kwargs.pop("pad_token_id", None)
    self.bos_token_id = kwargs.pop("bos_token_id", None)
    self.eos_token_id = kwargs.pop("eos_token_id", None)

    # Generation parameters exclusive to encoder-decoder models
    self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
    self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

    # Assistant generation
    self.is_assistant = False
    self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
    self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "constant")
    self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", 0.4)
    self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
    self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
    self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
    ## assistant generation for different tokenizers, the windows size for assistant/target model
    self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", 10)
    self.target_lookbehind = kwargs.pop("target_lookbehind", 10)

    # Performance
    self.compile_config = kwargs.pop("compile_config", CompileConfig())
    self.disable_compile = kwargs.pop("disable_compile", False)
    # Wild card
    self.generation_kwargs = kwargs.pop("generation_kwargs", {})

    # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
    # interface.
    self._from_model_config = kwargs.pop("_from_model_config", False)
    self._commit_hash = kwargs.pop("_commit_hash", None)
    self.transformers_version = kwargs.pop("transformers_version", __version__)

    # Additional attributes without default values
    if not self._from_model_config:
        # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
        # model's default configuration file
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    # Validate the values of the attributes
    self.validate(is_init=True)

def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
    """
    Returns the generation mode triggered by the [`GenerationConfig`] instance.

    Arg:
        assistant_model (`PreTrainedModel`, *optional*):
            The assistant model to be used for assisted generation. If set, the generation mode will be
            assisted generation.

    Returns:
        `GenerationMode`: The generation mode triggered by the instance.
    """
    # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
    # property and part of the `__repr__`
    if self.constraints is not None or self.force_words_ids is not None:
        generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
    elif self.num_beams == 1:
        if self.do_sample is False:
            if (
                self.top_k is not None
                and self.top_k > 1
                and self.penalty_alpha is not None
                and self.penalty_alpha > 0
            ):
                generation_mode = GenerationMode.CONTRASTIVE_SEARCH
            else:
                generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = GenerationMode.SAMPLE
    else:
        if self.num_beam_groups > 1:
            generation_mode = GenerationMode.GROUP_BEAM_SEARCH
        elif self.do_sample is True:
            generation_mode = GenerationMode.BEAM_SAMPLE
        else:
            generation_mode = GenerationMode.BEAM_SEARCH

    # Assisted generation may extend some generation modes
    if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.ASSISTED_GENERATION
        else:
            raise ValueError(
                "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                "is only supported with Greedy Search and Sample."
            )

    # DoLa generation may extend some generation modes
    if self.dola_layers is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.DOLA_GENERATION
        else:
            raise ValueError(
                "You've set `dola_layers`, which triggers DoLa generate. Currently, DoLa generate "
                "is only supported with Greedy Search and Sample."
            )
    
    # VDAHM generation
    if self.vdahm_layers is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.VDAHM_GENERATION
        else:
            raise ValueError(
                "You've set `vdahm_layers`, which triggers VDAHM generate. Currently, VDAHM generate "
                "is only supported with Sampling and Search."
            )

    return generation_mode


def _vdahm_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    image_ranges = getattr(generation_config, 'vdahm_image_ranges', None)

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    model_kwargs_tc = None
    
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if model_kwargs_tc is None:
            input_ids_tc = input_ids.clone()
            model_kwargs_tc = copy.deepcopy(model_kwargs)

            if image_ranges and image_ranges[0][0] > 0:
                model_kwargs_tc['pixel_values'] = None
                model_kwargs_tc['image_sizes'] = None
                model_kwargs_tc.pop('image_sizes', None)
              
                for start, end in image_ranges:
                    if start - 1 < input_ids_tc.size(1):
                        input_ids_tc[:, start-1] = pad_token_id
                    
                    if end + 1 < input_ids_tc.size(1):
                        input_ids_tc[:, end + 1] = pad_token_id

                for start, end in image_ranges:
                    if start - 1 < model_kwargs_tc["attention_mask"].size(1):
                        model_kwargs_tc["attention_mask"][:, start-1] = 0
                    
                    if end + 1 < model_kwargs_tc["attention_mask"].size(1):
                        model_kwargs_tc["attention_mask"][:, end + 1] = 0
            
            model_kwargs_tc = self._get_initial_cache_position(input_ids_tc, model_kwargs_tc)

            model_inputs_tc = self.prepare_inputs_for_generation(input_ids_tc, **model_kwargs_tc)

            # forward pass to store the attn_outputs of the text contrast inputs
            outputs_tc = self(**model_inputs_tc, return_dict=True)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        outputs = self(**model_inputs, return_dict=True)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits.clone()[:, -1, :].float()

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        
        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:  # argmax
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        # stop when each sentence is finished
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


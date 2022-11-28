# This script includes utils to extend the generation behavior of whisper so as to obtain confidence scores

from jiwer import wer
import math
import numpy as np
import os
import unicodedata
import re
import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation_utils import *
from transformers.tokenization_utils_base import to_py_obj
import types
from typing import Callable, Iterable, List, Optional, Tuple, Union
from whisper.normalizers import EnglishTextNormalizer

# set paths for confidence related files
predictions_confidence_path = os.path.join(os.getcwd(), 'predictions_confidence')
# create folders if they do not already exist
if not os.path.exists(predictions_confidence_path): os.makedirs(predictions_confidence_path)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WhisperWithConfidenceScores(WhisperForConditionalGeneration):
    """
    Wraps WhisperForConditionalGeneration with a new method : self.generate_with_confidence_scores that outputs tokens and token logits
    Original methods are under python3.9/site-packages/transformers/generation_utils.py with transformers 4.24.0
    """

    @torch.no_grad()
    def generate_with_confidence_scores(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int],
                                        Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[
            int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        **model_kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head. The method supports the following
        generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

            - *greedy decoding* by calling [`~generation_utils.GenerationMixin.greedy_search`] if `num_beams=1` and
              `do_sample=False`.
            - *contrastive search* by calling [`~generation_utils.GenerationMixin.contrastive_search`] if
              `penalty_alpha>0.` and `top_k>1`
            - *multinomial sampling* by calling [`~generation_utils.GenerationMixin.sample`] if `num_beams=1` and
              `do_sample=True`.
            - *beam-search decoding* by calling [`~generation_utils.GenerationMixin.beam_search`] if `num_beams>1` and
              `do_sample=False`.
            - *beam-search multinomial sampling* by calling [`~generation_utils.GenerationMixin.beam_sample`] if
              `num_beams>1` and `do_sample=True`.
            - *diverse beam-search decoding* by calling [`~generation_utils.GenerationMixin.group_beam_search`], if
              `num_beams>1` and `num_beam_groups>1`.
            - *constrained beam-search decoding* by calling
              [`~generation_utils.GenerationMixin.constrained_beam_search`], if `constraints!=None` or
              `force_words_ids!=None`.

        <Tip warning={true}>

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
        defined in the model's config (`config.json`) which in turn defaults to the
        [`~modeling_utils.PretrainedConfig`] of the model.

        </Tip>

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in
                the prompt.
            max_new_tokens (`int`, *optional*):
                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            min_length (`int`, *optional*, defaults to `model.config.min_length` or 10 if the config does not set any value):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `model.config.do_sample` or `False` if the config does not set any value):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to `model.config.num_beams` or 1 if the config does not set any value):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to `model.config.temperature` or 1.0 if the config does not set any value):
                The value used to module the next token probabilities.
            penalty_alpha (`float`, *optional*, defaults to `model.config.penalty_alpha` or None if the config does not set any value):
                The values balance the model confidence and the degeneration penalty in contrastive search decoding.
            top_k (`int`, *optional*, defaults to `model.config.top_k` or 50 if the config does not set any value):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to `model.config.top_p` or 1.0 if the config does not set any value):
                If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
                `top_p` or higher are kept for generation.
            typical_p (`float`, *optional*, defaults to `model.config.typical_p` or 1.0 if the config does not set any value):
                The amount of probability mass from the original distribution to be considered in typical decoding. If
                set to 1.0 it takes no effect. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
            repetition_penalty (`float`, *optional*, defaults to `model.config.repetition_penalty` or 1.0 if the config does not set any value):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*, defaults to `model.config.pad_token_id`):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*, defaults to `model.config.bos_token_id`):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*, defaults to `model.config.eos_token_id`):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to `model.config.length_penalty` or 1.0 if the config does not set any value):
                Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent
                to the sequence length, which in turn is used to divide the score of the sequence. Since the score is
                the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences,
                while `length_penalty` < 0.0 encourages shorter sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to `model.config.no_repeat_ngram_size` or 0 if the config does not set any value):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to `model.config.encoder_no_repeat_ngram_size` or 0 if the config does not set any value):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[List[int]]`, *optional*, defaults to `model.config.bad_words_ids`):
                List of token ids that are not allowed to be generated. In order to get the token ids of the words that
                should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
                add_special_tokens=False).input_ids`.
            force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
                List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple
                list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`,
                this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081),
                where one can allow different forms of each word.
            num_return_sequences(`int`, *optional*, defaults to `model.config.num_return_sequences` or 1 if the config does not set any value):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
                as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (`int`, *optional*, defaults to `model.config.num_beam_groups` or 1 if the config does not set any value):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to `model.config.diversity_penalty` or 0.0 if the config does not set any value):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown. This feature is intended for advanced users.
            renormalize_logits (`bool`, *optional*, defaults to `False`):
                Whether to renormalize the logits after applying all the logits processors or warpers (including the
                custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
                score logits are normalized but some logit processors or warpers break the normalization.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown. This feature is intended for advanced users.
            constraints (`List[Constraint]`, *optional*):
                 Custom constraints that can be added to the generation to ensure that the output will contain the use
                 of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
            output_attentions (`bool`, *optional*, defaults to `model.config.output_attentions` or `False` if the config does not set any value):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `model.config.output_hidden_states` or `False` if the config does not set any value):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `model.config.output_scores` or `False` if the config does not set any value):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `model.config.return_dict_in_generate` or `False` if the config does not set any value):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*, defaults to `model.config.forced_eos_token_id`):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            exponential_decay_length_penalty (`tuple(int, float)`, *optional*, defaults to `model.config.exponential_decay_length_penalty`):
                This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
                generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates
                where penalty starts and `decay_factor` represents the factor of exponential decay
            suppress_tokens  (`List[int]`, *optional*, defaults to `model.config.suppress_tokens`):
                A list of tokens that will be supressed at generation. The `SupressTokens` logit processor will set
                their log probs to `-inf` so that they are not sampled.
            begin_suppress_tokens  (`List[int]`, *optional*, defaults to `model.config.begin_suppress_tokens`):
                A list of tokens that will be supressed at the begining of the generation. The `SupressBeginTokens`
                logit processor will set their log probs to `-inf` so that they are not sampled.
            forced_decoder_ids (`List[List[int]]`, *optional*, defaults to `model.config.forced_decoder_ids`):
                A list of pairs of integers which indicates a mapping from generation indices to token indices that
                will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always
                be a token of index 123.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

        Examples:

        Greedy Decoding:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> prompt = "Today I believe we can finally"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        >>> # generate up to 30 tokens
        >>> outputs = model.generate(input_ids, do_sample=False, max_length=30)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today I believe we can finally get to the point where we can make a difference in the lives of the people of the United States of America.\n']
        ```

        Multinomial Sampling:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> prompt = "Today I believe we can finally"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        >>> # sample up to 30 tokens
        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.generate(input_ids, do_sample=True, max_length=30)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today I believe we can finally get rid of discrimination," said Rep. Mark Pocan (D-Wis.).\n\n"Just look at the']
        ```

        Beam-search decoding:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> sentence = "Paris is one of the densest populated areas in Europe."
        >>> input_ids = tokenizer(sentence, return_tensors="pt").input_ids

        >>> outputs = model.generate(input_ids, num_beams=5)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Paris ist eines der dichtesten besiedelten Gebiete Europas.']
        ```"""
        # 0. Validate the `.generate()` call
        self._validate_model_class()
        self._validate_model_kwargs(model_kwargs.copy())

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if pad_token_id is not None and torch.sum(inputs_tensor[:, -1] == pad_token_id) > 0:
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        if max_length is None and max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to "
                f"{self.config.max_length} (`self.config.max_length`). Controlling `max_length` via the config is "
                "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
                "using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
                "`max_new_tokens`."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None

        is_contrastive_search_gen_mode = (
            top_k is not None and top_k > 1 and do_sample is False and penalty_alpha is not None and penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )

        if num_beam_groups > num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
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

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            forced_decoder_ids=forced_decoder_ids,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )
        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search_with_confidence_scores(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:

            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing contrastive search."
                )

            return self.contrastive_search(
                input_ids,
                top_k=top_k,
                penalty_alpha=penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError(
                    "`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            if typical_p is not None:
                raise ValueError(
                    "Decoder argument `typical_p` is not supported with beam groups.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` needs to be greater than 1 for constrained generation.")

            if do_sample:
                raise ValueError(
                    "`do_sample` needs to be false for constrained generation.")

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError(
                    "`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if constraints is not None:
                final_constraints = constraints

            if force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {force_words_ids}."
                    )

                if not isinstance(force_words_ids, list) or len(force_words_ids) == 0:
                    typeerror()

                for word_ids in force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0)
                                for token_id in token_ids)
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

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def greedy_search_with_confidence_scores(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init tokens probs to NaNs for all padding tokens already in input_ids
        tokens_probs = torch.tensor([math.nan * input_ids.shape[0]])

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # compute prob of next_tokens
            all_next_tokens_probs = nn.functional.softmax(
                next_tokens_scores, dim=-1)
            next_tokens_probs = torch.max(all_next_tokens_probs)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + \
                    pad_token_id * (1 - unfinished_sequences)

            # update generated ids, generated_probs, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            tokens_probs = torch.cat(
                [tokens_probs, next_tokens_probs[None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids, tokens_probs


def batch_decode_processor(self, *args, **kwargs):
    """
    This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
    refer to the docstring of this method for more information.
    """
    return self.tokenizer.batch_decode(*args, **kwargs)


def batch_decode_tokenizer(
    self,
    sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
    probs: Union[List[float], "torch.Tensor"],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
) -> List[Tuple[str, List[str], List[float]]]:
    """
    Convert a list of lists of token ids into a list of strings by calling decode.
    Args:
        sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
            List of tokenized input ids. Can be obtained using the `__call__` method.
        skip_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to remove special tokens in the decoding.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
            Whether or not to clean up the tokenization spaces.
        kwargs (additional keyword arguments, *optional*):
            Will be passed to the underlying model specific decode method.
    Returns:
        `List[str]`: The list of decoded sentences.
    """
    return [
        self.decode(
            seq,
            probs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        for seq in sequences
    ]


def decode(
    self,
    token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
    token_probs: Union[float, List[float], "torch.Tensor"],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
) -> Tuple[str, List[str], List[float]]:
    """
    Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
    tokens and clean up tokenization spaces.
    Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.
    Args:
        token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
            List of tokenized input ids. Can be obtained using the `__call__` method.
        skip_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to remove special tokens in the decoding.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
            Whether or not to clean up the tokenization spaces.
        kwargs (additional keyword arguments, *optional*):
            Will be passed to the underlying model specific decode method.
    Returns:
        `str`: The decoded sentence.
    """
    # Convert inputs to python lists
    token_ids = to_py_obj(token_ids)

    return self._decode(
        token_ids=token_ids,
        token_probs=token_probs.tolist(),
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **kwargs,
    )


def _decode(
    self,
    token_ids: Union[int, List[int]],
    token_probs: Union[float, List[float]],
    skip_special_tokens: bool = False,
    normalize: bool = False, **kwargs
) -> Tuple[str, List[str], List[float]]:
    self._decode_use_source_tokenizer = kwargs.pop(
        "use_source_tokenizer", False)

    filtered_tokens, filtered_probs = self.convert_ids_to_tokens(
        token_ids, token_probs, skip_special_tokens=skip_special_tokens)

    # To avoid mixing byte-level and unicode for byte-level BPT
    # we need to build string separately for added tokens and byte-level tokens
    # cf. https://github.com/huggingface/transformers/issues/1133
    sub_texts = []
    current_sub_text = []
    probs = []

    for index, token in enumerate(filtered_tokens):
        if skip_special_tokens and token in self.all_special_ids:
            continue
        if token in self.added_tokens_encoder:
            raise NotImplementedError
        else:
            current_sub_text.append(token)
            probs.append(filtered_probs[index])
    if current_sub_text:
        sub_texts.append(self.convert_tokens_to_string(current_sub_text))

    text = "".join(sub_texts)

    if normalize:
        raise NotImplementedError
    else:
        # current_sub_text is list of final string tokens
        return text, current_sub_text, probs


def convert_ids_to_tokens(
    self,
    ids: Union[int, List[int]],
    probs: Union[float, List[float]],
    skip_special_tokens: bool = False
) -> Union[str, List[str]]:
    """
    Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
    added tokens.
    Args:
        ids (`int` or `List[int]`):
            The token id (or token ids) to convert to tokens.
        skip_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to remove special tokens in the decoding.
    Returns:
        `str` or `List[str]`: The decoded token(s).
    """
    if isinstance(ids, int):
        if ids in self.added_tokens_decoder:
            return self.added_tokens_decoder[ids]
        else:
            return self._convert_id_to_token(ids)
    tokens = []
    filtered_probs = []
    for ids_index, index in enumerate(ids):
        index = int(index)
        if skip_special_tokens and index in self.all_special_ids:
            continue
        if index in self.added_tokens_decoder:
            filtered_probs.append(probs[ids_index])
            tokens.append(self.added_tokens_decoder[index])
        else:
            filtered_probs.append(probs[ids_index])
            tokens.append(self._convert_id_to_token(index))
    return tokens, filtered_probs

# Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string with GPT2 -> Whisper


def convert_tokens_to_string(self, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    text = "".join(tokens)
    text = bytearray([self.byte_decoder[c]
                     for c in text]).decode("utf-8", errors=self.errors)
    return text


# Model loader
def load_whisper_with_confidence_scores(path: str, lang: str):
    """
    load and return wav2vec tokenizer and model from huggingface
    """
    processor = WhisperProcessor.from_pretrained(
        path, language=lang, task="transcribe")

    # modify relevant methods to retreive probs for unskipped tokens
    processor.batch_decode = types.MethodType(
        batch_decode_processor, processor)
    processor.tokenizer.batch_decode = types.MethodType(
        batch_decode_tokenizer, processor.tokenizer)
    processor.tokenizer.decode = types.MethodType(decode, processor.tokenizer)
    processor.tokenizer._decode = types.MethodType(
        _decode, processor.tokenizer)
    processor.tokenizer.convert_ids_to_tokens = types.MethodType(
        convert_ids_to_tokens, processor.tokenizer)
    processor.tokenizer.convert_tokens_to_string = types.MethodType(
        convert_tokens_to_string, processor.tokenizer)

    model = WhisperWithConfidenceScores.from_pretrained(path).to(device)

    return processor, model


def custom_normalizer(text, lang):
    """
    normalizing procedures based on appendix C, Whisper OpenAI paper
    language tokens based on https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
    """
    if lang == 'en':
        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    else:
        # removes [] and () as well as content in-between -- will not work for non-standard brackets, eg: <> or （）, etc
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        text = unicodedata.normalize("NFKC", text)
        ch_text = []
        for ch in text:
            if unicodedata.category(ch)[0] not in ('M', 'P', 'S'):
                ch_text.append(ch)
            else:
                ch_text.append(' ')
        text = ''.join(ch_text)
        text = text.lower()
    # set up for character error rate for languages w/o spaces between words
    if lang in ('zh', 'ja', 'th', 'lo', 'my'):
        text = ' '.join(text)
        # remove spaces between consecutive numbers
        text = re.sub('(?<=\d) (?=\d)', '', text)
    return re.sub(' +', ' ', text)


def map_to_pred_and_confidence_scores(batch, processor, model, lang):
    # read soundfile
    sampling_rate = batch.features["audio"].sampling_rate
    # recover input features
    input_features = processor(
        batch["audio"][0]["array"], sampling_rate=sampling_rate, return_tensors="pt").input_features
    # specify language of audio sample_rate
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang, task="transcribe")
    # generate logits and decode directly
    generated_ids, token_probs = model.generate_with_confidence_scores(
        inputs=input_features.to(device))
    transcription_with_probs = processor.batch_decode(
        generated_ids, token_probs, skip_special_tokens=True)
    batch['string_pred'] = [custom_normalizer(
        transcription_with_probs[0][0], lang)]
    batch['tokens_pred'] = [transcription_with_probs[0][1]]
    batch['probs_tokens_pred'] = [transcription_with_probs[0][2]]
    batch['ground_truth'] = [processor.tokenizer._normalize(unicodedata.normalize("NFKC", batch['transcription'][0]))]
    batch['wer'] = [wer(batch['string_pred'][0], batch['ground_truth'][0])]

    return batch


def html_display_confidence(prediction_dataset, rows_ids):
    """
    Compute html string with confidence color per token, ground truth and wer
    """

    final_text = ""

    def cstr(s, color='black'):
        return "<text style=color:{}>{}</text>".format(color, s)

    def map_float_rgb(f, m, M):
        rgb = 'rgb({},{},0)'.format(int(255 * (1 - ((f - m) / (M - m)))), int(255 * (f - m) / (M - m)))
        return rgb

    for row_index in rows_ids:
        tokens = prediction_dataset[row_index]['tokens_pred']
        probs_tokens = prediction_dataset[row_index]['probs_tokens_pred']


        min_prob = min(probs_tokens)
        max_prob = max(probs_tokens)


        final_text += "prediction &nbsp  &nbsp :  " + "".join([cstr(s=tokens[idx].replace('Ġ',' '), color=map_float_rgb(probs_tokens[idx], min_prob, max_prob)) for idx in range(len(tokens))]) + "<br>"
        final_text += "ground truth : " + prediction_dataset[row_index]['raw_transcription'] + "<br>"
        final_text += "WER" + 7 * " &nbsp" + ": " + str(round(100 * prediction_dataset[row_index]['wer'], 1)) + "%<br><br>"

    return final_text

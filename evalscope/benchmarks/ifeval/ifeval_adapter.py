from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.api.registry import get_aggregation, get_metric

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='ifeval',
        pretty_name='IFEval',
        description=
        'IFEval is a benchmark for evaluating instruction-following language models, focusing on their ability to understand and respond to various prompts. It includes a diverse set of tasks and metrics to assess model performance comprehensively.',  # noqa: E501
        tags=[Tags.INSTRUCTION_FOLLOWING],
        dataset_id='opencompass/ifeval',
        subset_list=['default'],
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='',
    )
)
class IFEvalAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        prompt = record.get('prompt', '')
        message_list = [ChatMessageUser(content=prompt)]

        return Sample(input=message_list, target='', metadata=record)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from evalscope.benchmarks.ifeval.utils import process_results

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata
        try:
            # Process results using the existing ifeval utility
            results = process_results(doc, [filtered_prediction])
            score.value.update(results)

            # Set main score name
            score.main_score_name = 'prompt_level_strict'

        except Exception as e:
            logger.error(f'Error calculating ifeval metrics: {e}')
            score.value = {}

        for metric in self.metric_list:
            if metric in ['prompt_level_strict', 'inst_level_strict', 'prompt_level_loose', 'inst_level_loose']:
                continue
            try:
                if isinstance(metric, str):
                    metric_name = metric
                    metric_scorer = get_metric(metric)  # Get metric implementation from registry
                    metric_func = metric_scorer()  # Instantiate the metric scorer
                elif isinstance(metric, dict):
                    metric_name = list(metric.keys())[0]
                    metric_cls = get_metric(metric_name)
                    metric_func = metric_cls(**metric[metric_name])
                if metric_name in ['tokens_num', 'think_num']:  # Initialize with parameters
                    metric_score = metric_func(
                        prediction=original_prediction,
                        reference=[],
                    )
                else:
                    metric_score = metric_func(
                        prediction=filtered_prediction,
                        reference=reference,
                    )
                score.value[metric_name] = metric_score
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                score.value[metric_name] = 0
                score.metadata[metric_name] = f'error: {str(e)}'

        return score

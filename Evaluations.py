from trulens.core import Metric,Selector
import numpy as np
import os
from trulens.providers.litellm import LiteLLM

provider = LiteLLM(
    model_engine="groq/openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
)

#Answer Relevance
answer_rel=Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

#Context Relevance
context_rel=Metric(
    implementation=provider.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question":Selector.select_record_input(),
        "context":Selector.select_context(collect_list=False)
    },
    agg=np.mean
)

#Groundedness
groundedness=Metric(
    implementation=provider.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source":Selector.select_context(collect_list=True),
        "statement":Selector.select_record_output()
    }
)




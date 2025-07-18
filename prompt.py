
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

new_template = "you are a helpfulrealestate agent. so act like that and  answer the user's question" + template
PROMPT = PromptTemplate(template=new_template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
from typing import Type, List

from cat.mad_hatter.decorators import hook
from cat.factory.embedder import EmbedderSettings
from langchain_community.embeddings import JinaEmbeddings
from pydantic import ConfigDict


class JinaEmbedderConfig(EmbedderSettings):
    jina_api_key: str
    model_name: str='jina-embeddings-v2-base-en'
    _pyclass: Type = JinaEmbeddings

    model_config = ConfigDict(
        json_schema_extra = {
            "humanReadableName": "Jina embedder",
            "description": "Jina embedder",
            "link": "https://jina.ai/embeddings/",
        }
    )


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed.append(JinaEmbedderConfig)
    return allowed
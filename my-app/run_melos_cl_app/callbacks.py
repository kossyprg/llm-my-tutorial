import chainlit as cl

from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document

class BaseCustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self, msg: cl.Message):
        AsyncCallbackHandler.__init__(self)
        self.msg = msg

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        for idx, d in enumerate(documents):
            source_name = f"参照情報 {idx+1}"
            content = self._arrange_reference_info(d)
            self.msg.elements.append(
                cl.Text(content=content, name=source_name, display="side")
            )
    
    def _arrange_reference_info(self, document: Document):
        ref_info =  f"{document.page_content}\n\n"
        ref_info += f"**著者**: {document.metadata['author']}\n\n"
        ref_info += f"**リンク**: [{document.metadata['link']}]({document.metadata['link']})\n\n"
        ref_info += f"**類似度スコア**: {document.metadata['score']:.2f}"
        return ref_info

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, cast

import google.generativeai as genai
import matplotlib.pyplot as plt
import qdrant_client
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    set_global_handler,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.query_engine.multi_modal import _get_image_and_text_nodes
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageDocument, ImageNode, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from PIL import Image

from .preprocess import VideoMetadata

set_global_handler("simple")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini = GeminiMultiModal(model_name="models/gemini-pro-vision")


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break
    plt.show()


def video_to_index(
    data_dir: Path,
    db_client: qdrant_client.QdrantClient,
    video_id: str,
    seconds_per_frame: int = 5,
) -> MultiModalVectorStoreIndex:
    text_store = QdrantVectorStore(
        client=db_client, collection_name=f"{video_id}_text_collection"
    )
    image_store = QdrantVectorStore(
        client=db_client, collection_name=f"{video_id}_image_collection"
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    img_documents = SimpleDirectoryReader(str(data_dir / "img")).load_data()
    for img_doc in img_documents:
        if isinstance(img_doc, ImageDocument) and img_doc.image_path:
            # Include the timestamp in the image document.
            frame_idx = int(img_doc.image_path.split("-")[-1].split(".")[0])
            seconds = frame_idx * seconds_per_frame
            m, s = divmod(seconds, 60)
            img_doc.metadata["timestamp"] = f"{m:02d}:{s:02d}"

    text_documents = SimpleDirectoryReader(str(data_dir / "transcription")).load_data()
    all_documents = img_documents + text_documents

    index = MultiModalVectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5"),
        show_progress=True,
    )
    # index.storage_context.persist(persist_dir="./storage")
    return index


def history_list_to_text(history: List[List[str]], limit=10):
    recent_history = history[:-limit]
    history_str = ""
    for message in recent_history:
        history_str += f"User: {message[0]}\n" f"Assistant: {message[1]}\n"
    return history_str


def build_image_context_str(image_nodes):
    image_context_str = ""
    for i, image_node in enumerate(image_nodes):
        timestamp = image_node.node.metadata["timestamp"]
        image_context_str += f"Image {i} timestamp: {timestamp}\n"
    return image_context_str


def chat(
    query_bundle: QueryBundle | str,
    history: List[List[str]],
    index: MultiModalVectorStoreIndex,
    video_metadata: VideoMetadata,
    return_context: bool = False,
) -> RESPONSE_TYPE:
    history_str = history_list_to_text(history)
    MULTI_QA_TEMPLATE_STR = (
        f'You are responsible to answer the questions from a user about a video called "{video_metadata["title"]}.\n"'
        "You have the following components at hands:\n"
        "1. The first image: the image is from the user, COMPARE it with others image to answer the query.\n"
        "2. Other images: they are cuts from the videos, use them to compare with the first image.\n"
        "Other images information is below.\n"
        "---------------------\n"
        "{img_context_str}\n"
        "---------------------\n"
        "3. Video transcription: also use the video transcription as context information to answer the query.\n"
        "Transcription is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Use the components to correctly answer the query.\n\n"
        "User: {query_str}\n"
        "Assistant: "
    )
    DEFAULT_QA_TEMPLATE_STR = (
        f'You are given the images and transcription from a video called "{video_metadata["title"]}".'
        "Context informatixon is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, try your best answer the query.\n"
        'If the query is not related to the video, kindly say "I can"t answer your question".\n'
        "User: {query_str}\n"
        "Assistant: "
    )
    DEFAULT_IMG_TEMPLATE_STR = (
        "Given the first image as the base image, use other images to "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    multi_input_qa_tmpl = PromptTemplate(
        MULTI_QA_TEMPLATE_STR, prompt_type=PromptType.QUESTION_ANSWER
    )
    text_qa_tmpl = PromptTemplate(
        DEFAULT_QA_TEMPLATE_STR, prompt_type=PromptType.QUESTION_ANSWER
    )
    img_qa_tmpl = PromptTemplate(
        DEFAULT_IMG_TEMPLATE_STR, prompt_type=PromptType.QUESTION_ANSWER
    )

    if isinstance(query_bundle, QueryBundle) and query_bundle.image_path:
        query_engine = index.as_query_engine(
            llm=gemini,
            text_qa_template=multi_input_qa_tmpl,
            image_qa_template=img_qa_tmpl,
            similarity_top_k=3,
            image_similarity_top_k=1,
        )
        query_engine = cast(SimpleMultiModalQueryEngine, query_engine)

        print("Retrieving using text...")
        text_nodes = query_engine.retrieve(query_bundle)
        text_nodes_ids = [node.node_id for node in text_nodes]
        print("Retrieving using image...")
        img_nodes = query_engine._retriever.image_to_image_retrieve(
            query_bundle.image_path
        )
        img_nodes = [node for node in img_nodes if node.node_id not in text_nodes_ids]
        img_nodes, _ = _get_image_and_text_nodes(img_nodes)
        nodes = img_nodes + text_nodes

        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join([r.get_content() for r in text_nodes])
        img_context_str = build_image_context_str(image_nodes)

        fmt_prompt = query_engine._text_qa_template.format(
            context_str=context_str,
            query_str=query_bundle.query_str,
            img_context_str=img_context_str,
        )
        image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        context_str = "\n\n".join([r.get_content() for r in text_nodes])
        img_context_str = build_image_context_str(image_nodes)

        fmt_prompt = query_engine._text_qa_template.format(
            context_str=context_str,
            query_str=query_bundle.query_str,
            img_context_str=img_context_str,
        )
        query_img_node = ImageNode(image_path=query_bundle.image_path)
        # Use the user's image.
        img_documents = [query_img_node]
        # Use the retrieved images.
        img_documents += [image_node.node for image_node in image_nodes]
        llm_response = query_engine._multi_modal_llm.complete(
            prompt=fmt_prompt, image_documents=img_documents
        )
        response = Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
        )

    else:
        query_engine = index.as_query_engine(
            llm=gemini,
            text_qa_template=text_qa_tmpl,
            image_qa_template=img_qa_tmpl,
            similarity_top_k=3,
            image_similarity_top_k=3,
        )
        query_engine = cast(SimpleMultiModalQueryEngine, query_engine)
        response = query_engine.query(query_bundle)

    print(f"Query:\n{query_bundle}\n Assistant:\n{str(response)}")

    if return_context:
        for text_node in response.metadata["text_nodes"]:
            display_source_node(text_node, source_length=2000)
        plot_images([n.metadata["file_path"] for n in response.metadata["image_nodes"]])

        print(
            "Retrieval:",
            len(response.metadata["image_nodes"]),
            "images, ",
            len(response.metadata["text_nodes"]),
            "context.",
        )

    return response


def _chat(
    query_str: QueryBundle | str,
    history: List[List[str]],
    index: MultiModalVectorStoreIndex,
    video_metadata: VideoMetadata,
    return_context: bool = False,
) -> RESPONSE_TYPE:
    history_str = history_list_to_text(history)

    DEFAULT_QA_TEMPLATE_STR = (
        f'You are given the images and transcription from a video called "{video_metadata["title"]}".'
        "Context informatixon is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, try your best answer the query.\n"
        'If the query is not related to the video, kindly say "I can"t answer your question".\n'
        f"{history_str}"
        "User: {query_str}\n"
        "Assistant: "
    )
    qa_tmpl = PromptTemplate(DEFAULT_QA_TEMPLATE_STR)
    query_engine = index.as_query_engine(
        llm=gemini,
        text_qa_template=qa_tmpl,
        similarity_top_k=3,
        image_similarity_top_k=3,
    )

    response = query_engine.query(query_str)

    # chat_bot = index.as_chat_engine(
    #     chat_mode=ChatMode.CONTEXT,
    #     llm=gemini,
    #     text_qa_template=qa_tmpl,
    #     similarity_top_k=5,
    #     image_similarity_top_k=3,
    #     memory=ChatMemoryBuffer.from_defaults(chat_history=history, token_limit=4096)
    # )

    print(f"Query:\n{query_str}\n Assistant:\n{str(response)}")
    print(
        "Retrieval:",
        len(response.metadata["image_nodes"]),
        "images, ",
        len(response.metadata["text_nodes"]),
        "context.",
    )
    if return_context:
        for text_node in response.metadata["text_nodes"]:
            display_source_node(text_node, source_length=2000)

        plot_images([n.metadata["file_path"] for n in response.metadata["image_nodes"]])

    return response

"""
    @Desc   :
    @Time   :2026/9/7 18:47
    @Author :爱吃肯德基
"""
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_use_md_splitter():
    headers_to_split_on = [
        ("#", "标题 1"),
        ("##", "标题 2"),
        ("###", "标题 3"),
        ("####", "标题 4"),
        ("#####", "标题 5"),
        ("######", "标题 6"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False,
        strip_headers=True,
    )
    file_path = r'D:\Python project\Shopkeeper_Brain\output\hak180产品安全手册\hak180产品安全手册_new.md'
    document = Path(file_path).read_text(encoding='utf-8')
    split_docs = md_splitter.split_text(document)
    for split_doc in split_docs:
        if split_doc.metadata:
            # 直接用 " > " 连接所有标题值（有多少层就拼多少层）
            header_prefix = "[" + " _ ".join(split_doc.metadata.values()) + "] "
        else:
            header_prefix = ""

        split_doc.page_content = header_prefix + '\n' + split_doc.page_content
        print(split_doc.page_content)
        print('------------------------------------------------------------------------------')

    print(f'总共{len(split_docs)}个doc')

    print('-------------调用RecursiveCharacterTextSplitter--------------')
    rc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "！", "。"],
        chunk_size=600,
        chunk_overlap=40
    )
    rc_docs = rc_splitter.split_documents(split_docs)
    for rc_doc in rc_docs:
        print(rc_doc)
        print('=============================================')
    print(f'总共{len(rc_docs)}个doc')


if __name__ == '__main__':
    split_use_md_splitter()

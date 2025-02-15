import time
import fitz
import json
import re
import base64
from dotenv import load_dotenv
from openai import OpenAI
import io
from openai import OpenAIError
from langchain.docstore.document import  Document
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ModelScopeEmbeddings


load_dotenv()
#提取pdf中文字
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text =""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text +=page.get_text('text')
    text = text.replace(' ','')
    text = text.replace('.','')
    # text = text.replace('\n','')
    return text

#找到书的目录
def find_first_directory(text):
    # 正则表达式匹配“目录”
    # pattern = r"(目录[\s\S]*?)(第一章[\s\S]*?)(?=\n第一章|\Z)"
    pattern = r"(目录[\s\S]*?)(第(?:1|一)(?:章|目|回)[\s\S]*?)(?=\s*第(?:1|一)(?:章|目|回)|\Z)"
    # 使用 re.search 查找匹配的内容
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 如果找到“目录”，返回其位置
        return match.end(), match.group()
    else:
        return None

def search_rag(query):
    embedding = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')
    faiss=FAISS.load_local('LLM.faiss',embedding,allow_dangerous_deserialization=True)
    result = faiss.similarity_search_with_score(query='梅毒',k=1)
    return result[0][0].page_content



client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("base_url"),
)
#加载大模型api
def load_api():
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("base_url")
    )
    return client
#提取大纲
def extract_outline(text):
    propmt =  ( f"你是一个处理文档的小助手，我将为你提供一本书的内容其中含有目录内容，你的任务是首先判断所有文本是否为目录文本如果不是你则需要提取目录文本"
                f"目录文本的提取方法：首先找到目录两个字 然后目录下的首先出现了章节名（其中含有序号或者章节标题等）然后记住该章名，然后再找到该章名再次出现时候，然后提取之间的内容"
                f'''**提取示例**：以下是文本
                目录
                内容简介...........................................................................................................................................................................2
                前言................................................................................................................................................................................3
                目录.................................................................................................................................................................................4
                计算机基础知识......................................................................................................................................................................6
                1.1计算机的发展...........................................................................................................................................................6
                1.1.1电子计算机的诞生........................................................................................................................................6
                1.1.2计算机发展的历程及未来趋势.......................................................................................................................8
                1.1.3计算机发展的新热点...................................................................................................................................11
                1.2计算机中信息的表示与存储...................................................................................................................................18
                1.3计算机病毒及防治..................................................................................................................................................30
                1.3.1计算机病毒的概念......................................................................................................................................30
                1.3.2计算机病毒的特点及分类............................................................................................................................30
                1.3.3计算机染毒的症状与防治措施.....................................................................................................................32
                第1章
                计算机基础知识
                1.1 计算机的发展
                1946年，世界上第一台电子计算机在美国宾夕法尼亚大学诞生。在短短的几十年里，电子
                计算机经历了几代演变，并迅速地渗透到人们生产和生活的各个领域，在科学计算、工程设计、
                数据处理以及人们的日常生活等领域发挥着巨大的作用。电子计算机被公认为20世纪最重大的
                工业革命成果之一。
                计算机是一种能够存储程序，并能按照程序自动、高速、精确地进行大量计算和信息处理
                的电子机器。科技的进步促使计算机的产生和迅速发展，而计算机的产生和发展又反过来促使
                科学技术和生产水平的提高。当今，电子计算机的发展和应用水平已经成为衡量一个国家科学
                技术水平和经济实力的重要标志
                '''
                f"对于以上文本找到第一个章节名为‘：计算机基础知识 然后再找到计算机基础知识 ，然后获取这中间的文本为目录内容："
                f'''计算机基础知识......................................................................................................................................................................6
                1.1计算机的发展...........................................................................................................................................................6
                1.1.1电子计算机的诞生........................................................................................................................................6
                1.1.2计算机发展的历程及未来趋势.......................................................................................................................8
                1.1.3计算机发展的新热点...................................................................................................................................11
                1.2计算机中信息的表示与存储...................................................................................................................................18
                1.3计算机病毒及防治..................................................................................................................................................30
                1.3.1计算机病毒的概念......................................................................................................................................30
                1.3.2计算机病毒的特点及分类............................................................................................................................30
                1.3.3计算机染毒的症状与防治措施.....................................................................................................................32'''

                f"1. **分级格式**：只分为两级"
                f"- 一级标题：在目录中作为大标题一般是第几章/回/目，但是有的时候会出现没有编号情况，则需要你的判断。输出前面加#"
                f"- 二级标题：在大标题下的相关内容，一般会出现第几节，或者有序号列如1.1这种格式的二级目录，三级目录则不要再作为二级标题了。输出时前面加 `-`，只输出节名，不需要加“第几节”，因此不需要输出编号。"
                f"2. **内容处理**："
                f"- 如果目录中带有“第几章/回/目”，则输出时保留“第几章/回/目”及其后面的内容。如果没有明确的章节编号，则直接输出内容，按层级归类。"
                f"- 严格按照给出的目录生成大纲。并且你要学会只能分析，将目录中一些无关紧要的名字剔除掉比如附录、前言、简介等，我们只要目录章节内容。"
                f"3. **输出格式**："
                f"- 仅生成大纲内容，禁止输出与目录无关的任何附加说明。"
                f"- 确保目录的完整性，不遗漏任何部分."
                f"以下是目录内容，请严格根据上述要求生成大纲：")
    client=load_api()
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': propmt},
                {'role': 'user', 'content': text}],
            presence_penalty=1.2,
            timeout=100
        )
        data_dict = json.loads(completion.model_dump_json())
        content = data_dict["choices"][0]["message"]["content"]
        cleaned_text = "\n".join(line for line in content.splitlines() if '大纲' not in line)
        return cleaned_text
    except OpenAIError as e:
        code = getattr(e, "code", None)  # 如果没有 'code' 属性，返回 None
        message = str(e)  # 获取完整的错误信息
        if code == 400 and  "data_inspection_failed" in message:
            # 处理内容审查错误
            return "内容审查错误："+str(message)
            # 可以重试请求或返回错误消息
        else:
            # 处理其他类型的错误
            return "错误："+str(e)
#通过文本提取知识点
def askQwen(article_content,theme):
    prompt = (f"您好！我正在处理一篇包含多个小标题的文章，并希望借助您的智能分析能力，进一步细化每个小标题下的内容，提取出更小一级的标题。"
              f"**输入格式**：文本 +已有的小标题"
              f"这些更小一级的标题应该能够概括每个小标题下各段落的主题或关键点。"
              f"请按照以下步骤操作 1、阅读并理解。 2、提取关键信息。 3、按照已有小标题下的内容生成更小一级的标题。4、按照已有的小标题对新的小标题进行分类。确保每个新的小标题只出现在一个最相关的节名下，避免重复。如果只有一个小节则不需要分类"
              f"只返回新的小标题，原来的小标题以#开头，新的子标题以-开头，请只输出标题，不要有其他输出"
              f"提取的新的小标题尽可能的不要出现在多个原来的小标题中，避免在不同的原标题中出现重复、多个新的小标题，如果出现原来的小标题中结构相同情况，则输出时尽可能不同"
              f"请你根据我提供的提取思路进行提取，不能根据其它思路进行提取"
              f"**提取思路**：首先根据已有的小标题的集合先从第一个小标题开始到第二个小标题在给出的文本中匹配两个标题之间的文本，然后只根据匹配到的文本进行更小一级标题的提取，放入到第一个已有小标题中。然后再根据第二个已有小标题到第三个已有小标题进行文本匹配，提取更小一级的标题放到第二个小标题中。直到所有的已有小标题进行提取完成。然后按照指定格式进行输出"
              )
    prompt_user = f"文章为：{article_content}，文章已有的小标题为：{theme}"
    client=load_api()
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': prompt_user}],
            presence_penalty=1.2,
            temperature=0.7,
            timeout=100
        )
        data_dict = json.loads(completion.model_dump_json())
        content = data_dict["choices"][0]["message"]["content"]
        return content
    except OpenAIError as e:
        code = getattr(e, "code", None)  # 如果没有 'code' 属性，返回 None
        message = str(e)  # 获取完整的错误信息
        if code == 400 and  "data_inspection_failed" in message:
            # 处理内容审查错误
            return "内容审查错误："+str(message)
            # 可以重试请求或返回错误消息
        else:
            # 处理其他类型的错误
            return "错误："+str(e)

def extract_ragtext(text,sencond_list):
    for i in range(len(sencond_list)):
        prompt = (
            f"现在给出你一段文本，请你严格依照我给出的文本进行如下的任务"
            f'我将会给出知识点所在的小节名以及该节的知识点，请你在我给出的文本中进行匹配，匹配到该节'
            f'然后你根据给出的知识点匹配到能对知识点解释的那段文本，然后你需要进行对该段文本进行提取摘要'
            f'摘要的要求需要能够合理的解释知识点然后你需要输出该知识点和摘要内容，我将回用来做RAG'
            f'每提炼出一个知识点的摘要则换行输出,输出的内容不需要任何额外符号'
            f"文本：{text}，节名:{sencond_list[i]['major_title']},知识点:{sencond_list[i]['sub_title']}"
        )
        client=load_api()
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'system', 'content': "你是一个专业的知识图谱专家，请帮我按下面的要求完成任务"},
                    {'role': 'user', 'content': prompt}
                ],
                presence_penalty=1.2,
                temperature=0.7,
                timeout=100
            )
            data_dict = json.loads(completion.model_dump_json())
            content = data_dict["choices"][0]["message"]["content"]
            with open('rag.txt','a',encoding='utf-8')as fp:
                fp.write(content)
        except OpenAIError as e:
            code = getattr(e, "code", None)  # 如果没有 'code' 属性，返回 None
            message = str(e)  # 获取完整的错误信息
            if code == 400 and  "data_inspection_failed" in message:
                # 处理内容审查错误
                print("内容审查错误："+str(message))
                # 可以重试请求或返回错误消息
            else:
                # 处理其他类型的错误
                return "错误："+str(e)



#提交给大模型，让大模型发现实体关系
def askQwen2(fir_knowledge_list,sec_knowledge_list):
    text1=search_rag(fir_knowledge_list)
    text2=search_rag(sec_knowledge_list)
    time.sleep(2)
    prompt = (
        "请基于以下给定的知识图谱实体及其定义，判断两个实体之间的关系，并按照要求的格式输出。"
        "我们提供了两个实体及其各自的定义，你需要根据定义理解两个实体的语义，判断它们之间是否存在以下四种关系之一："
        "1. **包含**：描述一个知识点是另一个知识点的子集或组成部分。例如，一个课程包含多个章节，章节包含多个知识点。这种关系用于明确知识点之间的层次结构。"
        "2. **前置**：表示学习某个知识点之前必须先掌握的知识点。这种关系体现了知识点之间的依赖性和先后顺序。例如，在学习“微积分”之前，需先掌握“极限”的概念。"
        " 3. **关联**：描述知识点之间的相互联系和逻辑关联。例如，“能量守恒”和“动量守恒”虽然是不同的概念，但在物理问题中常常相互关联。"
        " 4. **无**：当经过上述三种关系的比对后，两个实体之间不存在任何关系时选择此项。"
        " 请严格按照以下格式输出结果，并只输出结果内容，不添加额外解释或说明。"
        "**输出格式：**:实体1-关系-实体2"
        "输出格式：#主体-包含-主体 #主体-前置-主体。只输出提供的两个实体的关系不要有其余介绍"
        " **输入信息：**"
        "实体1：{knowledge1}"
        "实体2：{knowledge2}"
        "实体1的定义：{rag_text1}"
        "实体2的定义：{rag_text2}").format(knowledge1=fir_knowledge_list,knowledge2=sec_knowledge_list,rag_text1=text1,rag_text2=text2)
    client=load_api()
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': "你是一个专业的知识图谱专家，请帮我按下面的要求完成任务"},
            {'role': 'user', 'content': prompt}
        ],
        presence_penalty=1.2,
        temperature=0.7,
        timeout=100
    )
    data_dict = json.loads(completion.model_dump_json())
    content = data_dict["choices"][0]["message"]["content"]
    return content


#将一级标题和二级标题分别放入到不同的列表中
def chapter_list(text_list):
    first_title_list = []
    second_title_list = []
    middleware= []
    for id,line in enumerate(text_list.splitlines()):
        if line.startswith('#'):
            line = line.replace('#','')
            first_title_list.append(line)
            if int(id)!=0:
                second_title_list.append(middleware)
                middleware= []
        if line.startswith('-'):
            line = line .replace('-','')
            middleware.append(line)
    second_title_list.append(middleware)
    return first_title_list,second_title_list




#将提取的知识点做成json格式
def parse_res(input_string):
    # 初始化存储大标题和小标题的列表
    major_titles = []
    struct = []
    temp = []

    current_major_title = None
    current_minor_title = None

    # 分割字符串为行
    lines = input_string.split('\n')

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            if current_minor_title is not None:
                struct.append({
                    "major_title": major_titles[-1],
                    "sub_title": temp
                })
                temp=[]
            # 处理大标题
            current_major_title = stripped_line[1:].strip()
            major_titles.append(current_major_title)
            current_minor_title = None  # 重置小标题

        elif stripped_line.startswith('-'):
            # 处理小标题
            if current_major_title is not None:  # 确保已经有一个大标题
                current_minor_title = stripped_line[1:].strip()  # 去掉-并去除前后空格
                temp.append(current_minor_title)

    if current_minor_title is not None:
        struct.append({
            "major_title": major_titles[-1],
            "sub_title": temp
        })
    return struct

#构建每个章节的知识点，并按照json格式进行存储
def create_directory_structure(first_title_list,second_title_list,text):
    structure=[]
    for i, title in enumerate(first_title_list):
        # bug 考虑最后一章
        current_chapter = "".join(title.split())
        if len(first_title_list)!=1:
            if i == 0:
                # 第一章的模式，不考虑下一章
                next_chapter ="".join(first_title_list[i+1].split())
                pattern = r"^(.*?){}".format(next_chapter)
            elif i + 1 < len(first_title_list):
                next_chapter = "".join(first_title_list[i+1].split())
                # 所有章节之间的模式，包含换行符等任何内容
                pattern = r"{}[\s\S]*?(?={}|$)".format(current_chapter, next_chapter)
            else:
                # 最后一章的模式，只考虑到文本结尾
                pattern = r"{}[\s\S]*?$".format(current_chapter)
        else:
            pattern = r"{}[\s\S]*?$".format(current_chapter)
        # 提取出目标章节
        match = re.search(pattern, text, re.DOTALL)
        print(match.group())
        article_content = match.group().strip()
        # 根据小标题获取知识点
        if second_title_list[i]!=[]:
            sub_title = ''.join(second_title_list[i])
        else:
            sub_title = ['无标题']
        print("正在提取第{}章...".format(i))
        res = askQwen(article_content,sub_title)
        second_title_struct = parse_res(res)
        extract_ragtext(article_content,second_title_struct)
        data={
            "main_title": title,
            "sub_titles": second_title_struct
        }
        # with open('D:\\NLP\\neo4j-python程序\\教程知识图谱\\outline.json','a',encoding='utf-8')as fp:
        #     json.dump(data, fp,ensure_ascii=False)
        #     fp.write('\n')
        structure.append(data)
    return structure



def create_RAG():
    with open('rag.txt','r',encoding='utf-8')as fp :
        data=fp .read()
    print(data),exit()
#找到所有的知识点
def parse_struct(structure,second_title_list):
    # 解析struct
    second_account = 0 # 二级标题数量
    third_account = 0 # 三级标题数量
    second_title_list2 = [] # 存放二级标题
    third_title_list = [] # 存放三级标题

    for i in range(len(second_title_list)):
        second_account += len(structure[i]['sub_titles'])
        for j in range(len(structure[i]['sub_titles'])):
            third_account += len(structure[i]['sub_titles'][j]['sub_title'])
            second_title_list2.append(structure[i]['sub_titles'][j]['major_title'])
            for item in structure[i]['sub_titles'][j]['sub_title']:
                third_title_list.append(item)
    print("二级标题数量：{}，三级标题数量：{}".format(second_account,third_account))
    return third_title_list
def get_text(batch_size):
    with open('rag.txt','r',encoding='utf-8')as fp:
        data=''
        datalist=[]
        for i in fp:
            if i!='\n':
                data=data+i
            elif data!='':
                datalist.append(data)
                data=''
    vector_db = None
    batch=[]
    for line in datalist:
        line = line.strip()
        if line:  # 确保不处理空行
            batch.append(line)
            if len(batch) >= batch_size:
                vector_db = process_batch(batch, vector_db)
                batch = []  # 清空批次以释放内存

    # 处理最后一个批次（如果有）
    if batch:
        vector_db = process_batch(batch, vector_db)
    return vector_db

def process_batch(doc_batch, vector_db):
    embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')
    documents = [Document(metadata={'source': 'rag.txt'}, page_content=text) for text in doc_batch]
    temp_db = FAISS.from_documents(documents, embeddings)
    if vector_db is None:
        vector_db = temp_db
    else:
        vector_db.merge_from(temp_db)  # 合并新生成的索引到主索引
    return vector_db


#根据知识点进行实体关系构建
def get_knowledge_relation(knowledge_list):
    relation_list = []
    for i in range(len(knowledge_list)):
        fir_knowledge_list = knowledge_list[i]
        for j in range(int(i),len(knowledge_list)):
            sec_knowledge_list=knowledge_list[j]
            relation = askQwen2(fir_knowledge_list,sec_knowledge_list)
            with open('relation.txt','a',encoding='utf-8')as fp:
                fp.write(relation)
                fp.write('\n')

    return relation_list

def return_json_io(structure,relation_list):
    json_files = []
    for idx, data in enumerate(structure):
        json_content = json.dumps(data)
        json_file = io.BytesIO()
        json_file.write(json_content.encode('utf-8'))
        json_file.seek(0)
        encoded_json = base64.b64encode(json_file.read()).decode('utf-8')  # Base64 编码
        json_files.append((f"json_data_{idx+1}.json", encoded_json))

        # 将 TXT 文件转换为 Base64 编码的字符串
    txt_files = []
    for idx, txt_data in enumerate(relation_list):
        txt_file = io.BytesIO()
        txt_file.write(txt_data.encode('utf-8'))
        txt_file.seek(0)
        encoded_txt = base64.b64encode(txt_file.read()).decode('utf-8')  # Base64 编码
        txt_files.append((f"txt_data_{idx+1}.txt", encoded_txt))

    return json_files,txt_files


def method_collection(result,text):
    try:
        if result :
            if len(result[1])!=6000:
                outline= extract_outline(result[1])
            else:
                text=result[1][0:6000:1]
                outline= extract_outline(text)
            first_title_list,second_title_list = chapter_list(outline)
            print(first_title_list)
            print(second_title_list)
            text_tall= text[result[0]:]
            text_all = ''.join(text_tall.strip())
            if len(text_all)!=0:
                structure = create_directory_structure(first_title_list,second_title_list,text_all)
            else:
                text= ''.join(text.strip())
                structure = create_directory_structure(first_title_list,second_title_list,text)
            vector_db = get_text(batch_size=500)  # 根据实际情况调整批量大小
            vector_db.save_local('LLM.faiss')
            print('FAISS saved successfully!')
            knowledge_list = parse_struct(structure,second_title_list)
            relation_list=get_knowledge_relation(knowledge_list)
            return return_json_io(structure,relation_list)
        else:
            return '未找到该书的目录，请你传入一本教材'
    except :
        raise

if __name__=="__main__":
    path = 'D:\\NLP\\neo4j-python程序\\教程知识图谱\\大学计算机基础(1).pdf'
    text = extract_text_from_pdf(path)
    text = ''.join(line.strip()for line in text.splitlines())
    result = find_first_directory(text)
    reuslt_all =method_collection(result,text)
    print(reuslt_all)

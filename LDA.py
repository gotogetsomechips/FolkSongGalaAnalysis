# 导入必要的库
import pandas as pd
import numpy as np
import re
import os
import jieba
import jieba.analyse
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from datetime import datetime
from snownlp import SnowNLP
import sys
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import CoherenceModel
from itertools import combinations

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据加载与预处理
def load_and_preprocess_data(file_path):
    try:
        # 读取Excel文件，确保第一行为表头
        df = pd.read_excel(file_path, header=0)
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        sys.exit(1)
    # 检查是否包含必需的列
    required_columns = ['昵称', '评论', 'ip', '时间', '帖子链接', '文案', '点赞数', '回复数']
    for col in required_columns:
        if col not in df.columns:
            print(f"Excel文件中缺少列: {col}")
            sys.exit(1)
    # 打印数据的基本信息
    print(f"数据总行数（包含表头）：{len(df)}")
    print(f"数据列名：{df.columns.tolist()}")
    # 识别并移除包含重复表头的行
    header_mask = df['时间'].astype(str).str.strip() == '时间'
    if header_mask.any():
        print(f"发现 {header_mask.sum()} 行重复的表头，将被移除。")
        df = df[~header_mask]
    # 确保 '时间' 列为 datetime 类型，使用错误处理
    df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # 统计转换失败的行
    invalid_time_count = df['时间'].isna().sum()
    if invalid_time_count > 0:
        print(f"发现 {invalid_time_count} 行 '时间' 列无法转换，将被移除。")
        df = df.dropna(subset=['时间'])
    # 按时间排序
    df.sort_values(by='时间', inplace=True)
    # 数据清洗
    def clean_text(text):
        # 移除网址、表情符号、空格和非中文字符
        text = re.sub(r'http\S+', '', text)  # 移除网址
        text = re.sub(r'.*?', '', text)  # 移除表情符号
        text = re.sub(r'\s+', '', text)      # 移除空格
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 保留中文字符
        return text
    df['清洗后评论'] = df['评论'].astype(str).apply(clean_text)
    # 删除重复文本
    initial_count = len(df)
    df.drop_duplicates(subset=['清洗后评论'], inplace=True)
    duplicates_removed = initial_count - len(df)
    print(f"删除了 {duplicates_removed} 行重复的评论。")
    # 保存清洗后的数据
    df.to_excel('data/清洗后数据.xlsx', index=False)
    print("清洗后的数据已保存为 '清洗后数据.xlsx'。")
    return df
# 2. 分词与去停用词

def tokenize_and_remove_stopwords(df, stopwords_path, custom_stopwords_path):
    # 加载停用词
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())
    except Exception as e:
        print(f"加载停用词表时出错: {e}")
        sys.exit(1)
    # 加载自定义停用词
    try:
        with open(custom_stopwords_path, 'r', encoding='utf-8') as f:
            custom_stopwords = set(f.read().splitlines())
    except Exception as e:
        print(f"加载自定义停用词表时出错: {e}")
        sys.exit(1)
    all_stopwords = stopwords.union(custom_stopwords)
    # 分词
    def tokenize(text):
        return [word for word in jieba.cut(text, cut_all=False) if word not in all_stopwords and len(word) > 1]
    df['分词'] = df['清洗后评论'].apply(tokenize)
    print("分词和去停用词处理完成。")
    return df
# 3. 构建词典和语料库
def build_dictionary_corpus(df):
    # 创建词典
    dictionary = corpora.Dictionary(df['分词'])
    # 过滤极端词
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # 创建语料库
    corpus = [dictionary.doc2bow(text) for text in df['分词']]
    print("词典和语料库构建完成。")
    return dictionary, corpus
# 4. 确定最优主题数
def compute_perplexity(dictionary, corpus, texts, start=2, limit=11, step=1):
    perplexity_values = []
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        print(f"训练主题数为 {num_topics} 的LDA模型...")
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            passes=10
        )
        model_list.append(lda_model)
        
        # 计算困惑度
        perplexity = lda_model.log_perplexity(corpus)
        perplexity_values.append(perplexity)
        
        # 计算主题一致性
        coherence = evaluate_coherence(lda_model, texts, dictionary)
        coherence_values.append(coherence)
        
        print(f'主题数: {num_topics}, 困惑度: {perplexity:.3f}, 一致性: {coherence:.3f}')

    # 绘制双轴曲线
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()
    
    ax1.plot(range(start, limit, step), perplexity_values, 'b-', marker='o')
    ax2.plot(range(start, limit, step), coherence_values, 'r-', marker='s')
    
    ax1.set_xlabel('主题数')
    ax1.set_ylabel('困惑度', color='b')
    ax2.set_ylabel('主题一致性', color='r')
    plt.title('主题数量评估曲线')
    plt.grid(True)
    plt.savefig('visualizations/主题数量评估曲线.png')
    plt.close()
    return model_list, perplexity_values, coherence_values

# 5. 选择最优主题数并训练最终模型
def train_final_lda(dictionary, corpus, num_topics, alpha=50, eta=0.01):
    print(f"训练最终的LDA模型，主题数为 {num_topics}...")
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        alpha=[alpha]*num_topics,  # 对称先验
        eta=eta,                   # beta先验
        update_every=1,
        chunksize=100,
        passes=20,                 # 增加迭代次数
        per_word_topics=True
    )
    return lda_model
# 6. 输出LDA主题模型结果
def display_topics(lda_model, dictionary, num_words=10):
    print("LDA主题模型结果：")
    topics = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=num_words, formatted=False)
    for topic_no, topic in topics:
        print(f"主题 {topic_no +1}:")
        print(", ".join([word for word, _ in topic]))
        print("\n")
# 7. 构建关键词共现网络并可视化
# def visualize_cooccurrence(df, top_n=50):
#     """优化后的共现网络计算"""
#     cooccurrence = defaultdict(int)
    
#     # 优化后的组合计算
#     for tokens in df['分词']:
#         unique_tokens = list(set(tokens))  # 去重
#         for pair in combinations(unique_tokens, 2):
#             sorted_pair = tuple(sorted(pair))
#             cooccurrence[sorted_pair] += 1

#     # 选择出现频率较高的前N对
#     cooccurrence = dict(cooccurrence)
#     sorted_cooccurrence = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     # 构建图
#     G = nx.Graph()
#     for (word1, word2), weight in sorted_cooccurrence:
#         G.add_edge(word1, word2, weight=weight)
#     # 绘制图形
#     plt.figure(figsize=(12, 12))
#     pos = nx.spring_layout(G, k=0.15, iterations=20)
#     edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
#     nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
#     nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*0.1 for w in weights], alpha=0.5)
#     nx.draw_networkx_labels(G, pos, font_size=8)
#     plt.title('关键词共现网络')
#     plt.axis('off')
#     plt.show()

# LDAvis可视化
def visualize_with_ldavis(lda_model, corpus, dictionary):
    """使用pyLDAvis进行可视化"""
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'visualizations/lda_visualization.html')
    print("LDAvis可视化已保存为 lda_visualization.html")


# 主题一致性评估函数
def evaluate_coherence(lda_model, texts, dictionary):
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()
# 情感分析相关函数
def sentiment_analysis_snownlp(text):
    if not text.strip():
        return '中性'
    s = SnowNLP(text)
    sentiment_score = s.sentiments
    if sentiment_score > 0.6:
        return '正面'
    elif sentiment_score < 0.4:
        return '负面'
    else:
        return '中性'

def apply_sentiment_analysis(df):
    print("开始进行情感分析...")
    df['情感倾向'] = df['清洗后评论'].apply(sentiment_analysis_snownlp)
    print("情感分析完成。")
    return df

def visualize_sentiment(df, lda_model, dictionary):
    # 确保输出目录存在
    os.makedirs('visualizations', exist_ok=True)
    
    # 总体情感分布（添加数值标签）
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x='情感倾向', data=df, order=['正面', '负面', '中性'])
    plt.title('总体情感倾向分布')
    plt.xlabel('情感类别')
    plt.ylabel('数量')
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', 
            va='center', 
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=10
        )
    
    plt.savefig('visualizations/总体情感分布.png')
    plt.close()

    # 为文档分配主题
    df['主题分布'] = df['分词'].apply(lambda x: lda_model.get_document_topics(dictionary.doc2bow(x)))
    df['主要主题'] = df['主题分布'].apply(
        lambda x: sorted(x, key=lambda y: y[1], reverse=True)[0][0]+1 if x else None
    )

    # 不同主题下的情感分布
    plt.figure(figsize=(12,8))
    sns.countplot(x='情感倾向', hue='主要主题', data=df, palette='Set2', order=['正面', '负面', '中性'])
    plt.title('不同主题下的情感倾向分布')
    plt.xlabel('情感类别')
    plt.ylabel('数量')
    plt.legend(title='主题')
    plt.savefig('visualizations/主题情感分布.png')  # 保存图片
    plt.close()

def sentiment_over_time(df):
    os.makedirs('visualizations', exist_ok=True)
    
    df['日期'] = df['时间'].dt.date
    sentiment_time = df.groupby(['日期', '情感倾向']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(14,7))
    sentiment_time.plot(kind='line')
    plt.title('不同时间阶段的情感倾向分布')
    plt.xlabel('日期')
    plt.ylabel('评论数量')
    plt.legend(title='情感类别')
    plt.savefig('visualizations/时间情感趋势.png')  # 保存图片
    plt.close()

# 主函数
if __name__ == "__main__":
    # 设置文件路径
    data_file = 'data/总原始数据.xlsx'
    stopwords_file = 'data/哈工大停用词表.txt'
    custom_stopwords_file = 'data/自定义停用词表.txt'
    # 1. 加载与预处理数据
    df = load_and_preprocess_data(data_file)
    # 2. 分词与去停用词
    df = tokenize_and_remove_stopwords(df, stopwords_file, custom_stopwords_file)
    # 3. 构建词典和语料库
    dictionary, corpus = build_dictionary_corpus(df)
    # 4. 确定最优主题数
    # 这里设置主题数范围为2到11，步长为1
    model_list, perplexity_values, coherence_values = compute_perplexity(dictionary, corpus, df['分词'], start=2, limit=11, step=1)
    # 根据困惑度曲线选择最优主题数，例如选择困惑度最低的主题数
    optimal_index = np.argmin(perplexity_values)
    optimal_num_topics = 2 + optimal_index  # 因为start=2
    print(f'选择的最优主题数为: {optimal_num_topics}')
    # 5. 训练最终LDA模型
    lda_model = train_final_lda(dictionary, corpus, optimal_num_topics, alpha=50, eta=0.01)
    # 6. 输出LDA主题模型结果
    display_topics(lda_model, dictionary, num_words=10)
    # 7. 可视化关键词共现网络
    # visualize_cooccurrence(df, top_n=100)
    # 新增LDAvis可视化
    visualize_with_ldavis(lda_model, corpus, dictionary)
    # 8. 情感分析（使用SnowNLP）
    df = apply_sentiment_analysis(df)
    # 9. 可视化情感分析结果
    visualize_sentiment(df, lda_model, dictionary)
    # 10. 时间阶段情感分布
    sentiment_over_time(df)
    # 保存最终结果
    df.to_excel('data/最终分析结果.xlsx', index=False)
    print("最终分析结果已保存为 '最终分析结果.xlsx'。")


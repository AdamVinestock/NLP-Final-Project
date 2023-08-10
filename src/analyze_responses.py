import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_mean_ppx_instance(human_responses, machine_responses):
    """
    Each instance in the dataset has a unique "name" value with a different number of sentences for human and machine responses
    This method calculates the mean perplexity for each instance
    :param human_responses: df with responses
    :param machine_responses: df with responses
    :return: human and machine dataframe containing the mean perplexity for each instance
    """
    h_grouped_mean = human_responses.groupby('name')['response'].mean().reset_index()
    h_sorted_df = h_grouped_mean.sort_values(by='name', ascending=True)
    m_grouped_mean = machine_responses.groupby('name')['response'].mean().reset_index()
    m_sorted_df = m_grouped_mean.sort_values(by='name', ascending=True)
    return h_sorted_df, m_sorted_df

def calc_diff_ppx_instance(human_responses, machine_responses):
    """
    input: human and machine mean responses sorted my 'name' (samples)
    returns df holding name and the difference between human and machine responses
    Note: the larger the difference the better the context policy
    """
    diff_df = human_responses.copy()
    diff_df["response"] = human_responses["response"] - machine_responses["response"]
    return diff_df

def calc_sample_diff(human_path, machine_path):
    """
    input: paths of human and machine df holding responses for each sentence
    output: human-machine mean perplexity difference sample-wise
    """
    h_df = pd.read_csv(human_path)     # human responses df with perplexity over all sentences
    m_df = pd.read_csv(machine_path)   # machine responses df with perplexity over all sentences
    h_sorted_df, m_sorted_df = calc_mean_ppx_instance(h_df, m_df) # mean perplexity for each instance
    diff_df = calc_diff_ppx_instance(h_sorted_df, m_sorted_df)    # human - machine perplexity for each instance
    return diff_df['response'].mean()

def extract_info_from_path(path):
    """
    Extracts dataset name, author, model and context policy from the given path.
    """
    parts = path.split('/')
    name_parts = parts[-1].split('_')
    dataset_name = name_parts[0].replace('-',' ').capitalize()
    author = name_parts[1].replace('-',' ').capitalize()
    model = name_parts[2].replace('-',' ').capitalize()
    context_policy = name_parts[3].replace('-', ' ').capitalize()
    return dataset_name, author, model, context_policy

def create_hist(human_path, machine_path):
    """
    input: paths of human and machine csv holding responses for each sentence
    output: histogram of human and machine perplexity values
    """
    h_df = pd.read_csv(human_path)
    m_df = pd.read_csv(machine_path)
    dataset_name, author, model, context_policy = extract_info_from_path(human_path)
    bins = np.arange(min(h_df["response"].min(), m_df["response"].min()),
                     max(h_df["response"].max(), m_df["response"].max()),
                     0.1)
    plt.hist(h_df["response"], bins=bins, alpha=0.5, label='human text')
    plt.hist(m_df["response"], bins=bins, alpha=0.5, label='machine text')
    plt.title(f"Dataset - {dataset_name} with Context Policy - {context_policy}")
    plt.xlabel('log perplexity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

def compare_hist(human_path1, machine_path1, human_path2, machine_path2):
    """
    input: paths of human and machine csv holding responses for each sentence
    output: histograms of human and machine perplexity values
    """
    h_df1 = pd.read_csv(human_path1)
    m_df1 = pd.read_csv(machine_path1)
    h_df2 = pd.read_csv(human_path2)
    m_df2 = pd.read_csv(machine_path2)

    dataset_name1, author1, model1, context_policy1 = extract_info_from_path(human_path1)
    _, _, _, context_policy2 = extract_info_from_path(human_path2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(min(h_df1["response"].min(), m_df1["response"].min(),h_df2["response"].min(), m_df2["response"].min()),
                max(h_df1["response"].max(), m_df1["response"].max(), h_df2["response"].max(), m_df2["response"].max()),
                     0.1)
    axs[0].hist(h_df1["response"], bins=bins, alpha=0.5, label='human text')
    axs[0].hist(m_df1["response"], bins=bins, alpha=0.5, label='machine text')
    axs[0].set_title(f"Context policy - {context_policy1}")
    axs[0].set_xlabel('log perplexity')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(h_df2["response"], bins=bins, alpha=0.5, label='human text')
    axs[1].hist(m_df2["response"], bins=bins, alpha=0.5, label='machine text')
    axs[1].set_title(f"Context policy - {context_policy2}")
    axs[1].set_xlabel('log perplexity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.suptitle(f"Dataset - {dataset_name1}", fontsize=16)
    plt.tight_layout()
    plt.show()

def calc_diff(human_path, machine_path):
    """
    input: paths of human and machine csv's holding responses for each sentence
    output: (human - machine)/pooled_std perplexity difference
    """
    h_df, m_df = pd.read_csv(human_path), pd.read_csv(machine_path)
    h_mean, m_mean = h_df["response"].mean(), m_df["response"].mean()
    h_std, m_std = h_df["response"].std(), m_df["response"].std()
    pooled_std = np.sqrt((h_std**2 + m_std**2)/2)
    diff = (h_mean - m_mean)/pooled_std
    return diff








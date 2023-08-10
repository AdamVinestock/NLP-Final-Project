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

def calc_h_m_diff(human_path, machine_path):
    """
    input: paths of human and machine df holding responses for each sentence
    output: human-machine mean perplexity difference
    """
    h_df = pd.read_csv(human_path)     # human responses df with perplexity over all sentences
    m_df = pd.read_csv(machine_path)   # machine responses df with perplexity over all sentences
    h_sorted_df, m_sorted_df = calc_mean_ppx_instance(h_df, m_df) # mean perplexity for each instance
    diff_df = calc_diff_ppx_instance(h_sorted_df, m_sorted_df)    # human - machine perplexity for each instance
    return diff_df['response'].mean()

def create_hist(human_path, machine_path, title='Histograms of responses'):
    """
    input: paths of human and machine csv holding responses for each sentence
    output: histogram of human and machine perplexity values
    """
    h_df = pd.read_csv(human_path)
    m_df = pd.read_csv(machine_path)
    bins = np.arange(min(h_df["response"].min(), m_df["response"].min()),
                     max(h_df["response"].max(), m_df["response"].max()),
                     0.2)
    plt.hist(h_df["response"], bins=bins, alpha=0.5, label='human text')
    plt.hist(m_df["response"], bins=bins, alpha=0.5, label='machine text')
    plt.title(title)
    plt.xlabel('log perplexity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

def compare_hist(human_path1, machine_path1, human_path2, machine_path2, title='Histograms of responses'):
    """
    input: paths of human and machine csv holding responses for each sentence
    output: histogram of human and machine perplexity values
    """
    h_df1 = pd.read_csv(human_path1)
    m_df1 = pd.read_csv(machine_path1)

    h_df2 = pd.read_csv(human_path2)
    m_df2 = pd.read_csv(machine_path2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(min(h_df1["response"].min(), m_df1["response"].min(),h_df2["response"].min(), m_df2["response"].min()),
                max(h_df1["response"].max(), m_df1["response"].max(), h_df2["response"].max(), m_df2["response"].max()),
                     0.1)
    axs[0].hist(h_df1["response"], bins=bins, alpha=0.5, label='human text1')
    axs[0].hist(m_df1["response"], bins=bins, alpha=0.5, label='machine text1')
    axs[0].set_title('Dataset 1')
    axs[0].set_xlabel('log perplexity')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(h_df2["response"], bins=bins, alpha=0.5, label='human text2')
    axs[1].hist(m_df2["response"], bins=bins, alpha=0.5, label='machine text2')
    axs[1].set_title('Dataset 2')
    axs[1].set_xlabel('log perplexity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()







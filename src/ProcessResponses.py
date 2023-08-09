from many_atomic_detections import process_text, iterate_over_texts
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from src.dataset_loaders import (get_text_from_wiki_long_dataset,
                                 get_text_from_chatgpt_news_long_dataset,
                                 get_text_from_chatgpt_abstracts_dataset)
import pandas as pd

class ResponseClass():
    def __init__(self, dataset_name, model, model_name, tokenizer, context_policies, fixed_context, policy_names, from_sample = 0, to_sample =10):
        self.model = model
        self.model_name = model_name
        self.range = "[{}, {}]".format(from_sample, to_sample)
        self.dataset_name = dataset_name
        self.context_policies = context_policies
        self.context = fixed_context
        self.policy_names = policy_names
        self.from_sample = from_sample
        self.to_sample = to_sample
        self.sentence_detector = PerplexityEvaluator(model, tokenizer)
        self.human_dataset, self.machine_dataset = self.SplitDataset()
        self.parsers_list = self.CreateParsers()
        self.datasets_dict = {'human': self.human_dataset, 'machine': self.machine_dataset}
        self.human_responses, self.machine_responses = self.CalculatePerplexity()

    def SplitDataset(self):
        human_dataset = None
        machine_dataset = None
        if self.dataset_name == "wiki-intro-long":
            human_dataset = get_text_from_wiki_long_dataset(shuffle=False, text_field='human_text')
            machine_dataset = get_text_from_wiki_long_dataset(shuffle=False, text_field='machine_text')
        elif self.dataset_name == "news-chatgpt-long":
            human_dataset = get_text_from_chatgpt_news_long_dataset(shuffle=False, text_field='article')
            machine_dataset = get_text_from_chatgpt_news_long_dataset(shuffle=False, text_field='chatgpt')
        elif self.dataset_name == "ChatGPT-Research-Abstracts":
            human_dataset = get_text_from_chatgpt_abstracts_dataset(shuffle=False, text_field='real_abstract')
            machine_dataset = get_text_from_chatgpt_abstracts_dataset(shuffle=False, text_field='generated_abstract')

        truncated_human_dataset = human_dataset.select(range(self.from_sample, self.to_sample+1))
        truncated_machine_dataset = machine_dataset.select(range(self.from_sample, self.to_sample+1))

        return truncated_human_dataset, truncated_machine_dataset

    def CreateParsers(self):
        parsers = []
        for policy, context in zip(self.context_policies, self.context):
            parser = PrepareSentenceContext(context_policy = policy, context = context)
            parsers.append(parser)

        return parsers

    def CalculatePerplexity(self):
        # Perform log ppx calculation
        human_responses = []
        machine_responses = []
        i = 0
        for parser in self.parsers_list:
            for author in self.datasets_dict:  # human or machine
                csv_name = str(self.dataset_name)+"_"+str(author)+"_"+str(self.model_name)+"_"+self.policy_names[i]+"_"+self.range+'.csv'
                iterate_over_texts(self.datasets_dict[author], self.sentence_detector, parser, csv_name)
                if author == 'human':
                    df = pd.read_csv("Responses/"+csv_name)
                    human_responses.append(df)
                elif author == 'machine':
                    df = pd.read_csv("Responses/"+csv_name)
                    machine_responses.append(df)
            i += 1

        return human_responses, machine_responses

    def calc_mean_ppx_instance(self, human_responses, machine_responses):
        """
        Calculates the mean perplexity for each instance in the dataset
        :param human_responses: list of df's, each corresponding to a different context policy
        :param machine_responses: list of df's, each corresponding to a different context policy
        :return: list of df's, each dataframe contains the mean perplexity for each instance
        """

        human, machine = [], []

        for i, context_policy_df in enumerate(human_responses):
            h_grouped_mean = context_policy_df.groupby('name')['response'].mean().reset_index()
            sorted_df = h_grouped_mean.sort_values(by='name', ascending=True)
            human.append(sorted_df)


        for i, context_policy_df in enumerate(machine_responses):
            m_grouped_mean = context_policy_df.groupby('name')['response'].mean().reset_index()
            sorted_df = m_grouped_mean.sort_values(by='name', ascending=True)
            machine.append(sorted_df)

        return human, machine


    def calc_ppx_diff_chunk(self, human_responses, machine_responses):

        results = {}
        for i, policy in enumerate(human_responses):
            policy_results = (human_responses[i]['response'] - machine_responses[i]['response']).mean()
            results[self.policy_names[i]] = policy_results

        return results





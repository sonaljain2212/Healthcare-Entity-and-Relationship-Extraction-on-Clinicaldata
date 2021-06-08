# Healthcare-Entity-and-Relationship-Extraction-on-Clinicaldata

1 Introduction
Electronic health record (EHR) is a digital representation of a patient's paper chart. Although an
EHR system can include a patient's medical and medication histories, it is intended to go beyond
traditional clinical data gathered in a provider's oce. EHRs often contain the information regarding
the patient's medications, diagnosis, treatment plans, lab results and many other crucial information
that can be used to decide on further treatments. EHRs enable to share the information with other
health care providers such as medical imaging facilities, specialists, clinicians and anyone else involved
in the patient's care.
However, EHRs are highly unstructured data. They are lled without much regards to semantics,
punctuations, heterogeneous writing styles and interleaved medical jargon. To automate the process
of retrieving essential data eective and accurate natural language processing (NLP) techniques are
becoming increasingly important for use in computational data analysis. The NLP techniques we
focus on are Named Entity Recognition (NER) and Relation Extraction (RE). The NER allows us to
obtain the entities associated with an EHR such as the drug name, strength, route, frequency. Some
use cases of this are in identifying - the dosage associated with a drug, the frequency with which a
medication is usually prescribed and potentially the most crucial relationship - the potential adverse
eect that a drug can have.
In this project, we aim to perform these NLP tasks of NER and RE using existing biomedical deep
learning model, BioBERT. The dataset used in this project is the National NLP Clinical Challenges
(n2c2) [1] : Unstructured notes from the Research Patient Data Registry at Partners Healthcare and
the ADE Corpus. The n2c2 dataset consists of data that was collected from the MIMIC-III (Medical
InformationMart for Intensive Care III) clinical care database and consists of EHRs as text les along
with the corresponding annotations for the required entities.
First, we run the entire unannotated corpus through a LDA model to gure out segregation of
dierent topics in the dataset. And then, train BIOBert using the annotated dataset which yields
named entities, and we try to see if we obtain meaningful relations or patterns between the entities.
This is achieved using market basket analysis on the generated NERs. We see that LDA has been able
to uncover some meaningful topics from the data and also nd that market basket analysis is able to
uncover latent knowledge from the mined NERs.
2
2 Methods
2.1 Clustering
2.1.1 K Means
Kmeans algorithm [2] is an iterative algorithm that tries to partition the dataset into K pre-dened
distinct clusters where each data point belongs to only one group. It tries to make the intra-cluster
data points as similar as possible while also keeping the clusters as dierent (far) as possible. It
assigns data points to a cluster such that the sum of the squared distance between the data points
and the cluster's centroid is at the minimum. The less variation we have within clusters, the more
homogeneous the data points are within the same cluster.
The algorithm is as follows:
• Specify number of clusters K.
• Initialize centroids by rst shuing the dataset and then randomly selecting K data points for
the centroids without replacement.
• Keep iterating until there is no change to the centroids.
The objective function that we maximaze for K-means can be given by:
J =
Xm
i=1
XK
j=1
wikjjxi 􀀀 kjj2
where wik = 1 for data point xi if it belongs to cluster k; otherwise, wik = 0. Also, k is the
centroid of xi's cluster
2.1.2 t-SNE
t-SNE is unsupervised, randomized algorithm, used for visualization. It applies a non-linear dimen-
sionality reduction technique where the focus is on keeping the very similar data points close together
in lower-dimensional space. The t-SNE is a variation of Stochastic Neighbor Embedding (SNE) which
starts by converting the high-dimensional Euclidean distances between data points into conditional
probabilities that represent similarities. SNE minimizes the sum of Kullback-Leibler divergences over
all datapoints using a gradient descent method.
C = iKL(PijjQi)
= ijp(jji)log(
p(jji)
q(jji)
)
Where p(jji) is the actual probability function used in SNE and is given by
pjji =
exp(􀀀jjxj 􀀀 xijj)=22
i
K6=iexp(􀀀jjxj 􀀀 xijj)=22
i
Similarly, the conditional probability function chosen for the low-dimensional map (that we need to
optimize) is q(jji), given by
qjji =
exp(􀀀jjyj 􀀀 yijj)=22
i
K6=iexp(􀀀jjyj 􀀀 yijj)=22
i
3
2.2 Topic Modelling
Topic Modeling is a statistical technique for revealing the underlying semantic structure in large col-
lection of documents. It automatically analyzes text data to determine cluster words for a set of
documents. This is known as `unsupervised' machine learning because it doesn't require a predened
list of tags or training data that's been previously classied by humans. The most popular topic
modeling algorithms that contributed every sphere of text analysis in multiple domains includes La-
tent semantic analysis, Non-Negative Matrix Factorization, Probabilistic Latent semantic analysis,
and Latent Dirichlet Allocation. In this project, we used Latent Dirichlet allocation which is a gen-
erative probabilistic model for collections of discrete data such as text corpora. LDA is a three-level
hierarchical Bayesian model, in which each item of a collection is modeled as a nite mixture over an
underlying set of topics. Each topic is, in turn, modeled as an innite mixture over an underlying set
of topic probabilities.
For each document we have, (a) Distribution over topics
d  Dir()
Where Dir(.)is drawn from a uniform Dirichlet distribution with scaling parameter 
For each word in the document:
1. Draw a specic topic
zd;n  multi(d)
2. Draw a word
wd;n  z;d;n
The central inferential problem for LDA is determining the posterior distribution of latent variables
given the document.
P(; zjw; ; ) =
P(; z;wj; )
P(wj; )
2.3 Named Entity Recognition
The process of dening and categorizing key information (entities) in text is known as called entity
recognition (NER). Any term or group of words that repeatedly refers to the same thing is called an
object. Any detected entity is assigned to one of several categories. An NER machine learning (ML)
model, for example, could recognize the word \25 mg" in a text and classify it as a \Strength."
First step in NER is detecting a named entity. For example,\8.6 mg" is a string of 4 tokens that
represents one entity, namely the strength. Inside-outside-beginning tagging is a common way of
indicating where entities begin and end.
The next step requires the creation of entity categories. Some examples of categories include:
1. Frequency - bid (twice a day), qd (every day)
2. Drug - Vancomycin, Protonix
3. Route - po (by mouth), intravenous
A model involves training data to learn what is and is not a valid object, as well as how to categorize
them. The more task-relevant the training data, the more accurate the model will be at completing
the task. With the entities and the categories created we can use it to create a training and a test set
to train and validate an algorithm to see whether it can predict entities based on the token it is fed.
4
2.3.1 BioBERT
BioBERT, is a pre-trained language representation model for the biomedical domain. For the process
of pre-training and ne-tuning BioBERT, it was initialized with weights from BERT, which was pre-
trained on general domain corpora (English Wikipedia and BooksCorpus). BERT is a contextualized
word representation model that is pre-trained using bidirectional transformers and is based on a
masked language model. Previous language models were restricted to a two one-way language models
due to the nature of language modeling, where future words could not be seen. BERT employs a
masked language model that forecasts outcomes, at random, masked terms in a sequence and can thus
be used to learn bi-directional representation Then, BioBERT is pre-trained on biomedical domain
corpora (PubMed abstracts and PMC full-text articles) [3]. BioBERT can perform various tasks such
as NER, RE and Question-answering.
Figure 1: BioBERT Architecture
IOB2: In computational linguistics, the BIO / IOB format which stands for inside, outside,
beginning, is a standard tagging format for tagging tokens in a chunking task such as named-entity
recognition. The B- prex before a tag indicates that it is at the start of a chunk, while the I- prex
indicates that it is inside a chunk. When a tag is preceded by another of the same kind with no O
tokens between them, the B- tag is used. A token with an O tag does not belong to any person or
chunk.
For the sentence \Atenolol 25 milligrams po q day." We have the IOB2 labelling as given below:
Figure 2: IOB2 Annotation example
In IOB2 format every entity begins with the B tag compared to IOB scheme where only adjacent
entities of the same type will have B tag otherwise it starts with I tag
2.3.2 Med7
Med7 [4] is a named-entity recognition model for clinical natural language processing. The model is
trained to recognise seven categories: drug names, route, frequency, dosage, strength, form, duration.
5
The model was rst self-supervisedly pre-trained by predicting the next word, using a collection of 2
million free-text patients' records from MIMIC-III corpora and then ne-tuned on the named-entity
recognition task.The model achieved a lenient (strict) micro-averaged F1 score of 0.957 (0.893) across
all seven categories [5].
2.4 Market Basket Analysis
Market basket analysis (MBA) [6] is a set of statistical anity calculations that helps to study pur-
chasing patterns. In simplest terms, MBA shows what combinations of products most frequently occur
together in orders. These relationships can be used to develops relationships between the purchasing
patterns of dierent products. The following is the formal way of forming association rules:
fAig ! fCig
A collection of items is an itemset. The set of items on the left-hand side is the antecedent of the
rule, while the one to the right is the consequent. The probability that the antecedent event will occur
is the support of the rule. That simply refers to the relative frequency that an itemset appears in
transactions.
The probability that a customer will purchase a consequent on the condition of purchasing the an-
tecedent is referred to as the condence of the rule. Condence can be used for product placement
strategy and increasing protability.
The lift of the rule is the ratio of the support of the antecedents co-occurring with the consequent,
divided by the probability that the antecedents and consequent co-occur if the two are independent.
The following can be inferred from lift values:
• A lift greater than 1 suggests that the presence of the antecedent increases the chances that the
consequent will occur in a given transaction.
• Lift below 1 indicates that purchasing the antecedent reduces the chances of purchasing the
consequent in the same transaction.
• When the lift is 1, then purchasing the antecedent makes no dierence on the chances of pur-
chasing the consequent
The Apriori algorithm [6] is a commonly-applied technique in computational statistics that identies
itemsets that occur with a support greater than a pre-dened value (frequency) and calculates the
condence of all possible rules based on those itemsets. We will be using this algorithm for our analysis.
6

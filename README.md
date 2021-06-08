# Healthcare-Entity-and-Relationship-Extraction-on-Clinicaldata

1 Introduction
Electronic health record (EHR) is a digital representation of a patient's paper chart. Although an
EHR system can include a patient's medical and medication histories, it is intended to go beyond
traditional clinical data gathered in a provider's once. EHRs often contain the information regarding
the patient's medications, diagnosis, treatment plans, lab results and many other crucial information
that can be used to decide on further treatments. EHRs enable to share the information with other
health care providers such as medical imaging facilities, specialists, clinicians and anyone else involved
in the patient's care.
However, EHRs are highly unstructured data. They are filled without much regards to semantics,
punctuations, heterogeneous writing styles and interleaved medical jargon. To automate the process
of retrieving essential data effective and accurate natural language processing (NLP) techniques are
becoming increasingly important for use in computational data analysis. The NLP techniques we
focus on are Named Entity Recognition (NER) and Relation Extraction (RE). The NER allows us to
obtain the entities associated with an EHR such as the drug name, strength, route, frequency. Some
use cases of this are in identifying - the dosage associated with a drug, the frequency with which a
medication is usually prescribed and potentially the most crucial relationship - the potential adverse
effect that a drug can have.
In this project, we aim to perform these NLP tasks of NER and RE using existing biomedical deep
learning model, BioBERT. The dataset used in this project is the National NLP Clinical Challenges
(n2c2) [1] : Unstructured notes from the Research Patient Data Registry at Partners Healthcare and
the ADE Corpus. The n2c2 dataset consists of data that was collected from the MIMIC-III (Medical
InformationMart for Intensive Care III) clinical care database and consists of EHRs as text files along
with the corresponding annotations for the required entities.
First, we run the entire unannotated corpus through a LDA model to figure out segregation of
different topics in the dataset. And then, train BIOBert using the annotated dataset which yields
named entities, and we try to see if we obtain meaningful relations or patterns between the entities.
This is achieved using market basket analysis on the generated NERs. We see that LDA has been able
to uncover some meaningful topics from the data and also and that market basket analysis is able to
uncover latent knowledge from the mined NERs.

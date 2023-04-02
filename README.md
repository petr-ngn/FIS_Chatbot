# NLP Chatbot for FIS VŠE
**TEAM**:
   - **Lead Data Scientist & Developer**: [**Petr Nguyen**](https://www.linkedin.com/in/petr-ngn/)
   - **Data Scientist Support**: [Lukas Dolezal](https://www.linkedin.com/in/lukas-dolezal75/)
   - **Business Analysts**: [Peter Kachnic](https://www.linkedin.com/in/peterkachnic/), [Karolina Benkovicova](https://www.linkedin.com/in/karolina-benkovicova-460/), [Andrea Novakova](https://www.linkedin.com/in/andrea-novakova/), [Samuel Nagy](https://www.linkedin.com/in/samuel-nagy-a31b51113/), [Adrian Harvan](https://www.linkedin.com/in/adrian-harvan/)
   
Within the course __*Trends in business analytics II (4IT409)*__, supervised by [Filip Vencovsky, Ph.D.](https://www.linkedin.com/in/filipvencovsky/), we had to
1. conduct a business research regarding the current chatbot of Faculty of Informatics and Statistics at Prague University of Economics and Business (FIS VŠE), assess its features and propose improvements, and
2. implement our improvements in the deployment either using already existing pretrained chatbots or models or developing our own chatbot from scratch.

Particularly, we have our developed our custom chatbot from scratch using optimized neural network. We created our own intents of questions which were the most relevant to study at FIS VŠE, which were further preprocessed using NLTK (for tokenization) and Majka (for stemming and lemmatization of Czech terms) into bag of words based on which we developed a custom neural network in Keras which was further tuned using Bayesian Optimization.

Using such developed neural network, we then built a custom chatbot which we deployed as a ML web application using Flask - the interface was built using HTML and Javascript for user-friendly experience with the chatbot.



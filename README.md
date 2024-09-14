# LetsCopeOrg

First Part: letscope_data_fetching.py
* reads data from database and generates data from content/videos, quizzes, memberships.
* It also generates for one user the quiz he had submitted, the content he watched, the membership he is paying and store them.
* the data data is stored into folders into formats like csv or json.

Second Part: Letscope_chat_2.py
* it uses only LlamaIndex python package and OPENAI api key service.
* Data folders:
  * data_articles: docx documents given by letscope and made by professionals.
  * data_content: constains the csv file generated previously with video's data.
  * data_membership: contain the json of all membership.
  * data_quiz: contain the json of all quizzes.
  * data_user: contains all user folder with their json generated via letscope_data_fetching.py.
  * data_website: contain all html pages of the website letscope.org.
* For data_content, data_articles and data_website it creates a custom in memory vector database index and query engines:
  * for data data_content it adds to the index document the metadata of the video to be queried easier and post process if needed to retrieve names or link.
  * for data_articles,a full custom query engine was developed to use text embedding and keywords extraction by intersecting the both to find a better context for answering question about coping knowledge.
  * for data_website, every html files was load with an html loader and the query engine prompt was modified to adjust the responses.
* After it creates an openai agent with 3 tools:
  * a top k video recommendations based on the data content index. the query and the k parameters are made by the agent while conserving withe the user.
  * a query engine tool (rag) on the articles data.
  * a query engine tool (rag) on the website data.
* At the end it makes the agent and use it as a chat with its instruction in a propmt.


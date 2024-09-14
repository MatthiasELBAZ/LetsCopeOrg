# LetsCopeOrg

# First Part: letscope_data_fetching.py
* reads data from __database__ and generates data from content/videos, quizzes, memberships.
* It also generates for __one user__ the quiz he had submitted, the content he watched, the membership he is paying and store them.
* the data data is stored into folders into formats like __csv__ or __json__.

# Second Part: Letscope_chat_2.py
* it uses only __LlamaIndex__ python package and __OPENAI__ api key service.

## Data folders:
  * __data_articles__: docx documents given by letscope and made by professionals.
  * __data_content__: constains the csv file generated previously with video's data.
  * __data_membership__: contain the json of all membership.
  * __data_quiz__: contain the json of all quizzes.
  * __data_user__: contains all user folder with their json generated via letscope_data_fetching.py.
  * __data_website__: contain all html pages of the website letscope.org.

## Query Engines
* For data_content, data_articles and data_website it creates a __custom in memory vector database index and query engines__:
  * for data data_content it adds to the index document the __metadata__ of the video to be __queried easier__ and post process if needed to retrieve names or link.
  * for data_articles,a full custom query engine was developed to __use text embedding and keywords extraction__ by intersecting the both to find a better context for answering question about coping knowledge.
  * for data_website, every html files was load with an __html loader__ and the query engine __prompt__ was modified to adjust the responses.
 
## Agent Tools
* After it creates an openai agent with 3 tools:
  * a __top k video recommendations__ based on the data content index. the query and the k parameters are made by the agent while conserving withe the user.
  * a __query engine__ tool (rag) on the articles data.
  * a __query engine__ tool (rag) on the website data.
 
## Agent Chat
* At the end it makes the agent and use it as a chat with its __instruction in a propmt__.


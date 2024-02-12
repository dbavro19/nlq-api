#Using MarketData API - translate natural languege questions into API calls, and use the reults to answer the original question in context 

import requests
import boto3
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NLQ-API", page_icon=":tada", layout="wide")

#Headers
with st.container():
    st.header("Natural Languege Queries to API Call")
    st.subheader("Natural Languege Queries to API Calls")
    #st.title("Ask Questions about your ECS and ObjectScale system")

with st.container():
    st.write("---")
    userQuery = st.text_input("Ask A Question About A Finnancial Instrument (as long as it is Apple)")
    #userID = st.text_input("User ID")
    st.write("---")

bedrock = boto3.client('bedrock-runtime' , 'us-east-1')


def categorize_question(question):


     ##Setup Prompt
    prompt_data = f"""
Classify the User question into the most appropriate category based on the category description

categories:
Quotes - For getting real-time price quotes for a stock.
Earnings - For getting historical prices for a stock
N/A - If the question does not fit into any of the categories

user_question:
What are the company's main competitors 

Return only one of the categories (Candles, Quotes, Earnings, N/A, or Not Supported). No other text
Begin:
"""


    body = json.dumps({
        "inputText": prompt_data, 
        "textGenerationConfig":{
            "maxTokenCount":4096,
            "stopSequences":[],
            "temperature":0
            }
        }) 
    
    #Run Inference
    modelId = 'amazon.titan-tg1-large' # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'
    outputText = "\n"

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body.get('results')[0].get('outputText')


    #result=parse_xml(llmOutput, "category")

    #if result=="Not Supported":
    #    return "Not Supported"
    #elif result=="N/A":
    #    return "N/A"
    #elif result=="Candles":
    #    return "Candles"
    #elif result=="Quotes":
    #    return "Quotes"
    #elif result=="Earnings":
    #    return "Earnings"
    #else:
    #    return "Something Went Wrong"

    #print(llmOutput)

    return llmOutput


def parse_xml(xml, tag):
    temp=xml.split(">")
    
    tag_to_extract="</"+tag

    for line in temp:
        if tag_to_extract in line:
            parsed_value=line.replace(tag_to_extract, "")
            return parsed_value

def quote_question_to_API(question):
     ##Setup Prompt
    prompt_data = f"""
Human: 

You are an Financial Assistant that help translate user questions into API calls to get information about financial instruments 
Based on the provided question, format a valid API url based on the format provided including all required parameters
Include relevant optional parameters if the user's questions requires it. Do not include any optional parameters that are not needed to answer the user's question

<user_question>
{question}
</user_question>

<api_format>
https://api.marketdata.app/v1/stocks/quotes/(symbol)/
</api_format>

<required_request_parameters>
symbol: string

The company's ticker symbol. If no exchange is specified, by default a US exchange will be assumed. You may embed the exchange in the ticker symbol using the Yahoo Finance or TradingView formats. A company or securities identifier can also be used instead of a ticker symbol.

Ticker Formats: (TICKER, TICKER.EX, EXCHANGE:TICKER)

Company Identifiers: (CIK, LEI)

Securities Identifiers: (CUSIP, SEDOL, ISIN, FIGI)
</required_request_parameters>

<optional_request_parameters>
52week: boolean

Enable the output of 52-week high and 52-week low data in the quote output. By default this parameter is false if omitted.
</optional_request_parameters>



Return only one the formatted API call in an <api> xml tag. No other text

Assistant:

"""


    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":1000,
                 "temperature":0,
                 "top_k":250,
                 "top_p":0.5,
                 "stop_sequences":[]
                  }) 
    
    #Run Inference
    modelId = "anthropic.claude-instant-v1"  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body.get('completion')

    result=parse_xml(llmOutput, "api")

    #I may need a mapping file/looklup to tranlate company/instrument to valid ID

    return result



def call_API(url):

    url2 = str(url).strip()
    response = requests.request("GET", url2)

    return response.text

def get_answer(question, data):
         ##Setup Prompt
    prompt_data = f"""
You are an Financial Assistant that helps answer user questions about financial instruments using the payload of an API call
Based on the data provided, provide a succinct answer to the user's question
Use the provided API response information for reference to the data's format and attributes
Think through your answer and be as accurate as possible
Only use the provided data to answer the question, dont make assumptions

<user_question>
{question}
</user_question>


<response_information>
-s: string
Will always be ok when there is data for the symbol requested.

-symbol: array[string]
The symbol of the stock.

-ask: array[number]
The ask price of the stock.

-askSize: array[number]
The number of shares offered at the ask price.

-bid: array[number]
The bid price.

-bidSize: array[number]
The number of shares that may be sold at the bid price.

-mid: array[number]
The midpoint price between the ask and the bid.

-last: array[number]
The last price the stock traded at.

-change: array[number]
The difference in price in dollars (or the security's currency if different from dollars) compared to the closing price of the previous day.

-changepct: array[number]
The difference in price in percent compared to the closing price of the previous day.

-52weekHigh: array[number]
The 52-week high for the stock. This parameter is omitted unless the optional 52week request parameter is set to true.

-52weekLow: array[number]
The 52-week low for the stock. This parameter is omitted unless the optional 52week request parameter is set to true.

-volume: array[number]
The number of shares traded during the current session.

-updated: array[date]
The date/time of the current stock quote.
</response_information>

<data>
{data}
</data>



Return your answer to the user's question in a conversational manner, no other text

Begin:

"""


    body = json.dumps({
        "inputText": prompt_data, 
        "textGenerationConfig":{
            "maxTokenCount":4096,
            "stopSequences":[],
            "temperature":0
            }
        }) 
    
    
    #Run Inference
    modelId = 'amazon.titan-tg1-large' # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'
    outputText = "\n"

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body.get('results')[0].get('outputText')



    return llmOutput


#question = "What is the latest change percent of Apple's stock?"



ask=st.button("ASK!")
if ask:
    category = categorize_question(userQuery)

    if category=="Not Supported":
        
        st.write(category)

    elif category=="N/A":
        print(category)
        st.write(category)

    elif category=="Candles":
        print(category)
        st.write(category)

    elif category=="Quotes":
        with st.status("Processing Request", expanded=False, state="running") as status:
            status.update(label="Category: " + category + " Identified", state="running", expanded=False)
            st.write(":heavy_check_mark: Category: " + category)

            url = quote_question_to_API(userQuery)
            print(url)
            status.update(label="API URL Generated: " + url, state="running", expanded=False)
            st.write(":heavy_check_mark: API URL: " + url)

            api_results = call_API(url)
            print(api_results)
            status.update(label="API Call results: " + api_results, state="running", expanded=False)
            st.write(":heavy_check_mark: API results: " + api_results)

            df = pd.DataFrame(json.loads(api_results))
            df_transposed = df.transpose()

            status.update(label="DataFrame: " + str(df_transposed), state="running", expanded=False)
            #st.write(":heavy_check_mark: DataFrame: " + str(df_transposed))
            st.write(df_transposed)


            answer = get_answer(userQuery, df_transposed)
            status.update(label="Answer Generated: " + answer, state="complete", expanded=False)
            st.write(":heavy_check_mark: Answer: " + answer)

        st.write(answer)

    #json.loads(api_results)
    #df = pd.DataFrame(json.loads(api_results))
    #df_transposed = df.transpose()

    #answer2 = get_answer(question, df_transposed)

    elif category=="Earnings":
        print(category)
        st.write(category)
            
    else:
        print(category + " Is not a valid option")
        st.write(category)









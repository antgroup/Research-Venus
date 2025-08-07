import requests
import json,time
import sys
import json
import time
import io
import uuid
import subprocess
from transformers import AutoTokenizer
import yaml
import glob
import os
import json5
from tqdm import tqdm
from openai import OpenAI
import re
import http.client

def serper_google_search(
        query, 
        top_k = 10,
        region = 'us',
        lang = 'en',
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        print('data',data)
        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print("search success")
            return results
    except Exception as e:
        if depth < 512:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []
    
'''
vllm serve /root/mymodel --tensor-parallel-size 4 --port 3001 --api-key 123 --served-model-name mymodel
'''


serper_api_key = ''
llmclient = OpenAI(
    base_url="http://localhost:3001/v1", 
    api_key="123",
)
modelname = '/root/mymodel'

tokenizer = AutoTokenizer.from_pretrained(modelname)

system_prompt = r'''## Background information 
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with tools to help you answer the question:
- web_search: Search the web for relevant information from google. You should use this tool if the historical page contentis not enough to answer the question. Or last search result is not relevant to the question.

You don't have to answer the question now, but you should first think about the research plan or what to search next.

Your output format should be one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{"type": "function", "function": {"name": "web_search", "description": "Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search, which helps answer the question"}}, "example": {"name": "web_search", "arguments": {"query": "xxxx"}}, "uniqueItems": true}}}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''


def parse_response(response_contents, think: bool = True):
    results = []
    for i, content in enumerate(response_contents):
        if "<think>" in content and "<answer>" in content:
            if "</think>" not in content or "</answer>" not in content:
                results.append((True, "", ""))
            else:
                think = content.split("<think>")[1].split("</think>")[0]
                answer = content.split("<answer>")[1].split("</answer>")[0]
                results.append((True, think, answer))
        elif "<think>" in content and "<tool_call>" in content:
            if "</tool_call>" not in content or "</think>" not in content:
                results.append((True, "", ""))
            else:
                think = content.split("<think>")[1].split("</think>")[0]
                tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    tool_call = json.loads(tool_call)
                    results.append((False, think, tool_call))
                except Exception as e:
                    print(f"model tool call format error: {e}")
                    print(i, content)
                    results.append((True, "", ""))
        else:
            results.append((True, "", ""))
    return results


def execute_predictions(
        tool_call_list, total_number
    ) :
    
    query_contents = [{"idx": tool_call[0], "question": tool_call[1], "think": tool_call[2],
                       "tool_call": tool_call[3], "total_number":total_number} for tool_call in tool_call_list]
    tool_result = []
    for query_content in query_contents:
        result = ''
        try:
            if query_content['tool_call']["name"] == 'web_search':
                result = ant_search(query_content['tool_call']['arguments']['query'])
        except:
            import traceback
            print(f"{str(traceback.format_exc())}")
        query_content['content'] = result
        tool_result.append(query_content)
    return tool_result


def run_prompts(questions):
    ii = 0
    isfinish = 0
    messages_list = []
    for question in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        messages_list.append(messages)
    history = ''
    max_turns = 5
    activate_list = [i for i in range(len(messages_list))]
    for step in range(max_turns):
        activate_messages_list = [messages_list[i] for i in activate_list]
        if activate_list == []:
            break
        rollings_active = tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tokenize=False)
        rollings_active = [rolling + "<think>" for rolling in rollings_active]
        completion = llmclient.completions.create(
            model="mymodel",
            prompt=rollings_active,
            temperature=0.0,
            max_tokens=2024,
        )
        responses = ["<think>" + x['text'] for x in json.loads(completion.model_dump_json())['choices']]
        results = parse_response(responses, think=True)
        activate_list_copy = []
        tool_call_list = []
        for i in range(len(results)):
            if results[i][0]:
                messages_list[activate_list[i]].append({
                    "role": "assistant", 
                    "content": responses[i], 
                })
            else:
                activate_list_copy.append(activate_list[i])
                tool_call_list.append((activate_list[i], messages_list[activate_list[i]][1]["content"], results[i][1], results[i][2]))

        tool_call_list = execute_predictions(tool_call_list,len(messages_list))
        for i in range(len(tool_call_list)):
            messages_list[tool_call_list[i]['idx']].append(
                {
                    "role": "assistant", 
                    "content": "<think>" + tool_call_list[i]['think'] + "</think>", 
                    "tool_calls": [
                                    {
                                        "type": "function", 
                                        "function": tool_call_list[i]['tool_call']
                                    }
                                ]
                }
            )
            try:
                messages_list[tool_call_list[i]['idx']].append(
                    {
                        "role": "tool", 
                        "name": tool_call_list[i]['tool_call']['name'],
                        "content": tool_call_list[i]['content']
                    }
                )
            except:
                messages_list[tool_call_list[i]['idx']].append(
                    {
                        "role": "tool", 
                        "name": '',
                        "content": '返回格式错误，tool调用失败'
                    }
                )
        activate_list = activate_list_copy
    return messages_list

questions = ['In a prestigious British equestrian event known for its historical significance and economic influence in the UK, a figure famously associated with Christmas gift-giving secured a surprising victory in 1964. What is the name of this notable race where the unexpected triumph took place?']
results = run_prompts(questions)

for i in range(len(questions)):
    print('\n\n------')
    print(i)
    print(questions[i])
    result = results[i]
    for j in range(len(result)):
        print('---')
        print(result[j])
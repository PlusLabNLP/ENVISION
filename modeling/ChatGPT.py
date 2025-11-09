import openai
import csv, os, sys, re, string, json, glob, shutil, random, datetime, math, string
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import pprint
import backoff
import copy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
stoplist = set(stopwords.words('english'))

from ego4d_chatgpt_utils import SRLParser
srl_parser = SRLParser(srl_model="structured-prediction-srl-bert")

def get_srl_arg1(srl_parser, sent):
    srl_res = srl_parser.get_arg_by_tags(                                              
        sentence=sent,                                                             
        tags_of_interests=["ARG1"],                                                    
    )    
    if srl_res is not None:                                                            
        pprint.pprint(srl_res)                                                         
        return srl_res["ARG1"] 
    else:
        return None



class GPTSymbolic(object):
    def __init__(self, k_trials_ooc = 3, model="gpt-3.5-turbo", mode="run"):
        self.model = model
        self.mode = mode
        self.k_trials_ooc = k_trials_ooc
    
    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError,openai.error.ServiceUnavailableError,openai.error.APIError))
    def _generate(self, prompt):
        result = openai.ChatCompletion.create(
          model=self.model,
          messages=[
                {"role": "system", "content": prompt},
            ]
        )
        return result

    def _further_ground(self, sent, obj):
        skeleton = (
            "Please output an exact subpart of the sentence [{}] where the object [{}] is referred to. "
            "Please remove leading verbs in your answer. "
            "Please output only the exact sentence subpart and nothing else. "
            "Please make sure that you answer can be exactly found in the sentence [{}]. "
            "If question is not valid, please simply output [None] and nothing else."
            "Please do not output any explaination of your answer. "
        )

        prompt = skeleton.format(sent, obj, sent)
        ans = self._generate(prompt)

        if self.mode == "debug":
            print(prompt)
            print(ans)
            
        result = ans["choices"][0]["message"]["content"]
    
        return result

    def _extract_obj(self, sent):
        skeleton = (
            "From the action [{}], please extract: "
            "(a) The one single object that is passively being processed or manipulated "
            "(b) The one single tool used to manipulate (a), or if tool is non-existent then the single object/container that supports the action. "
            "Please do not contain the word 'hand' or 'finger' in (b). "
            "Please do not explain anything in you answer. "
            "Please only output one object for each part. "
            "If you cannot find either (a) or (b), please simply output 'None' for that field and print nothing else."
        )
        prompt = skeleton.format(sent)
        ans = self._generate(prompt)
        if self.mode == "debug":
            print(prompt)
            print(ans)
        try:
            result = {
                "gpt_responses" : ans["choices"][0]["message"]["content"],
                "ooc": ans["choices"][0]["message"]["content"].split("(a)")[1].split("(b)")[0].strip().lower(),
                "tool": ans["choices"][0]["message"]["content"].split("(b)")[1].strip().lower(),
            }
        except:
            result = {
                "gpt_responses" : ans["choices"][0]["message"]["content"],
                "ooc": "None",
                "tool": "None",
            }
        return result
    
    def _spatial(self, sent, ooc, tool):
        skeleton = (
            "From the first person view of the action {}, what should the positional "
            "relationship between {} and {} be? Please choose from: (1). largely overlapping, "
            "(2). slightly touching, (3). a short distance away from, (4). a long distance "
            "away from (output only one number)"
        )
        spatial_dict = {
            "1": "(1). largely overlapping",
            "2": "(2). slightly touching",
            "3": "(3). a short distance away from",
            "4": "(4). a long distance away from",
        }
        prompt = skeleton.format(sent, ooc, tool)
        ans = self._generate(prompt)
        if self.mode == "debug":
            print(prompt)
            print(ans)
        try:
            result = {
                "gpt_responses" : ans["choices"][0]["message"]["content"],
                "spatial": spatial_dict[[i for i in ans["choices"][0]["message"]["content"] if i.isdigit()][0]]
            }
        except:
            result = {
                "gpt_responses" : ans["choices"][0]["message"]["content"],
                "spatial": "(3). a short distance away from"
            }
        return result
    
    def _size(self, sent, ooc, tool):
        skeleton = (
            "From the first person view of the action {}, please compare the size of the objects {} "
            "and {} based on common sense. Which object is likely larger in size, please just print "
            "one of the options along with option number: (1). obj1, (2). obj2, or (3). similar"
        )
        size_dict = {
            "1": "(1). ooc is larger",
            "2": "(2). tool is larger",
            "3": "(3). similar",
        }
        prompt = skeleton.format(sent, ooc, tool)
        ans = self._generate(prompt)
        if self.mode == "debug":
            print(prompt)
            print(ans)
        try:
            result = {
                "gpt_responses" : ans["choices"][0]["message"]["content"],
                "size": size_dict[[i for i in ans["choices"][0]["message"]["content"] if i.isdigit()][0]]
            }
        except:
            result = {
            "gpt_responses" : ans["choices"][0]["message"]["content"],
            "size": None
        }
        return result
    
    def _state_change(self, sent, obj):
        skeleton = (
            "From the first person view of the action {}, would the object {} undergo significant "
            "change in its visual appearance? If so, please output [yes] in the first line and If not, "
            "please output [no] in the first line. If the answer is yes, then on the second line, please "
            "simply print one or two visually recognizable adjectives to describe the appearance of the "
            "object before the state change, on the third line, please simply print a few (<7) words "
            "that visually describe the object after the state change"
        )
        prompt = skeleton.format(sent, obj)
        ans = self._generate(prompt)
        if self.mode == "debug":
            print(prompt)
            print(ans)
        result = {
            "gpt_responses" : ans["choices"][0]["message"]["content"],
            "state_change": ans["choices"][0]["message"]["content"].split("\n")[0],
            "pre_state": None,
            "post_state": None,
        }
        if len(result["gpt_responses"].split("\n")) == 3:
            result["pre_state"] = result["gpt_responses"].split("\n")[1]
            result["post_state"] = result["gpt_responses"].split("\n")[2]
        return result
    
    
    def _define(self, obj):
        skeleton = "In one sentence, please define the object [{}]"
        prompt = skeleton.format(obj)
        ans = self._generate(prompt)
        if self.mode == "debug":
            print(prompt)
            print(ans)
        result = {
            "gpt_responses" : ans["choices"][0]["message"]["content"],
        }
        return result
    
    def get_gpt_response(self, sent, ooc = None, tool = None, obj = None, mode=None):
        if mode == "extract_obj":
            return self._extract_obj(sent)
        elif mode == "spatial":
            return self._spatial(sent, ooc, tool)
        elif mode == "size":
            return self._size(sent, ooc, tool)
        elif mode == "state_change":
            return self._state_change(sent, obj)
        elif mode == "definition":
            return self._define(obj)
        else:
            print("The specified mode is not defined!")
            raise

    
        
    def pipeline(self, sent):
        
        for i in range(self.k_trials_ooc):
            extract_obj_res = self.get_gpt_response(sent, mode="extract_obj")
            ooc = extract_obj_res["ooc"].lower().translate(str.maketrans('', '', string.punctuation)).strip()
            tool = extract_obj_res["tool"].lower().translate(str.maketrans('', '', string.punctuation)).strip()
            if not "none" in ooc:
                break
        
        if "none" in ooc:
            if self.mode == "debug":
                print("There is no ooc detected")
            results = {
                "original_sent": sent,
                "gpt_responses": {
                    "extract_obj": None,
                    "spatial": None,
                    "size": None,
                    "ooc_state_change": None,
                    "tool_state_change": None,
                },
                "parsed_responses": {
                    "ooc": None,
                    "tool": None,
                    "spatial": None,
                    "size" : None,
                    "ooc_state_change": {
                            "state_change": None,
                            "pre_state": None,
                            "post_state": None,
                        },
                    "tool_state_change": None,
                }
            }
            return results
        elif "none" in tool or tool not in sent:
            # here I added the case of tool not in sent b/c I don't want gpt to guess the tool
            if self.mode == "debug":
                print("There is an ooc but no tools is detected, checking ooc state change alone")

            if ooc not in sent:
                ooc = self._further_ground(sent, ooc)
            if ooc not in sent:
                ooc = get_srl_arg1(srl_parser, sent)

            ooc_state_change_res = self.get_gpt_response(sent, ooc, mode="state_change")
            results = {
                "original_sent": sent,
                "gpt_responses": {
                    "extract_obj": extract_obj_res["gpt_responses"],
                    "spatial": None,
                    "size": None,
                    "ooc_state_change": ooc_state_change_res["gpt_responses"],
                    "tool_state_change": None,
                },
                "parsed_responses": {
                    "ooc": ooc,
                    "tool": None,
                    "spatial": None,
                    "size" : None,
                    "ooc_state_change": {
                            "state_change": ooc_state_change_res["state_change"],
                            "pre_state": ooc_state_change_res["pre_state"],
                            "post_state": ooc_state_change_res["post_state"],
                        },
                    "tool_state_change": None,
                }
            }
        else:
            # in this case tool and ooc has to both exist and tool has to be in sent

            if self.mode == "debug":
                print("Both ooc and tool are detected, checking spatial, size and state change")
            
            # emperically, when ooc is very ungroundable, it is often better to use srl_arg1
            if ooc not in sent:
                ooc = self._further_ground(sent, ooc)
            if ooc not in sent:
                ooc = get_srl_arg1(srl_parser, sent)
            
            # emperically, when tool is very ungroundable, it is often better to use the ungroundable tool
            # ungroundable_tool = 0
            # if tool not in sent:
            #     ungroundable_tool = 1
            
            spatial_res = self.get_gpt_response(sent, ooc, tool, mode="spatial")
            size_res = self.get_gpt_response(sent, ooc, tool, mode="size")
            ooc_state_change_res = self.get_gpt_response(sent, obj = ooc, mode="state_change")
            tool_state_change_res = self.get_gpt_response(sent, obj = tool, mode="state_change")
            

            results = {
                "original_sent": sent,
                "gpt_responses": {
                    "extract_obj": extract_obj_res["gpt_responses"],
                    "spatial": spatial_res["gpt_responses"],
                    "size": size_res["gpt_responses"],
                    "ooc_state_change": ooc_state_change_res["gpt_responses"],
                    "tool_state_change": tool_state_change_res["gpt_responses"],
                },
                "parsed_responses": {
                    "ooc": ooc,
                    "tool": tool,
                    "spatial": spatial_res["spatial"],
                    "size" : size_res["size"],
                    "ooc_state_change": {
                        "state_change": ooc_state_change_res["state_change"],
                        "pre_state": ooc_state_change_res["pre_state"],
                        "post_state": ooc_state_change_res["post_state"],
                    },
                    "tool_state_change": {
                        "state_change": tool_state_change_res["state_change"],
                        "pre_state": tool_state_change_res["pre_state"],
                        "post_state": tool_state_change_res["post_state"],
                    },
                }
            }
        return results
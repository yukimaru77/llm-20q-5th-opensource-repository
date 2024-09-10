# Agent Alpha using alphabetical order/comparison and bisection.
# Optimal search, guaranteed to find the solution in 20 turns (log2(n_keywords)+1 upper bound)
# if only:
#   * the solution keyword is on the list of all words (below), not exceeding 2**19 items
#   * the other team player also plays Alpha, i.e. it is able to respond to alpha questions correctly
#
# If the solution keyword is not on the list, alpha search cannot guess the keyword, but still, the result may
# be useful when combined with other techniques.

# It's also possible to play passively as the Answerer without the handshake protocol. Although it's
# unlikely to see Questioners trying Alpha without the handshake. Anyway, all is needed is a regex
# matcher in front of the answering pipeline. See the implementation.

# Cheers, loh-maa


import random
import re
import string
def find_matching_indices(lst, target_lst):
    matching_indices = []

    for target in target_lst:
        if target in lst:
            matching_indices.append(lst.index(target))

    return sorted(matching_indices)
def find_bounds(lst, lower_bound, upper_bound):
    lower_index = -1
    upper_index = -1

    # Find the lower index
    for i, item in enumerate(lst):
        if item >= lower_bound:
            lower_index = i
            break

    # Find the upper index
    for i, item in enumerate(lst):
        if item >= upper_bound:
            upper_index = i - 1
            break
    else:
        upper_index = len(lst) - 1

    return lower_index, upper_index

def generate_question_start(letters):
    if len(letters) == 1:
        return f"Does the keyword start with the letter '{letters[0]}'?"
    else:
        joined_letters = "', '".join(letters[:-1])  # 最後の文字以外を 'で区切って結合
        last_letter = letters[-1]
        return f"Does the keyword start with one of the letters '{joined_letters}' or '{last_letter}'?"
def generate_end_question(letters):
    if len(letters) == 1:
        return f"Does the keyword end with the letter '{letters[0]}'?"
    else:
        joined_letters = "', '".join(letters[:-1])  # 最後の文字以外を 'で区切って結合
        last_letter = letters[-1]
        return f"Does the keyword end with one of the letters '{joined_letters}' or '{last_letter}'?"
def count_first_letter_occurrences(word_list,index):
    first_letter_counts = {}
    for word in word_list:
        if word:
            first_letter = word[index].lower()  # 大文字と小文字を区別しない場合はlower()を使用
            if first_letter in first_letter_counts:
                first_letter_counts[first_letter] += 1
            else:
                first_letter_counts[first_letter] = 1
    return first_letter_counts

def find_closest_half(n):
    #lstは2^n-1の値が入っているものを作成。n=1~20の場合の値が入っている
    lst = [2**i-1 for i in range(1,21)]
    half_n = n / 2
    closest_value = min(lst, key=lambda x: abs(x - half_n))
    if closest_value > half_n:
        return n - closest_value
    else:
        return closest_value

import re
import string
def normalize(s: str) -> str:
    t = str.maketrans("", "", string.punctuation)
    # 文字列の先頭の "the " を削除
    s = s.lower()
    if s.startswith("the "):
        s = s[4:]
    return s.translate(t).strip()
def _normalize(s: str) -> str:
    t = str.maketrans("", "", string.punctuation)
    return s.lower().replace("the", "").replace(" ", "").translate(t)

def compare_words(a, b) -> bool:
    a = _normalize(a)
    b = _normalize(b)
    if a == b:
        return True
    # don't check for plurals if string is too short
    if len(a) < 3 or len(b) < 3:
        return False
    # accept common plurals
    if a[-1] == "s" and a[:-1] == b:
        return True
    if b[-1] == "s" and a == b[:-1]:
        return True
    if a[-2:] == "es" and a[:-2] == b:
        return True
    if b[-2:] == "es" and a == b[:-2]:
        return True
    return False
def simple_question(query,keyward):
  #想定質問1 "letter"という文字と"begin"または"start"。またはfirst letterという文字
  #  "Does the keyword end with the letter 'e'?"
  if ("letter" in query.lower() and (("begin" in query.lower()) or ("start" in query.lower()))) or ("first letter" in query.lower()) or ("character of the keyword one of" in query.lower()):
    if "second word" in query.lower():
        keyward = keyward.split(" ")
        if len(keyward) >= 2:
            keyward = keyward[1]
        else:
            keyward = keyward[0] # 1文字しかない場合はそのまま
    # 2. 'で囲まれている領域または"で囲まれている領域を取得
    tmp_query = query
    if "and end with a letter" in query.lower():
       query = query.lower().split("and end with a letter")[0]
       query_end = tmp_query.lower().split("and end with a letter")[1]
    matches = re.findall(r"[\"'](.*?)[\"']", query)
    index = 0
    if "second character" in query.lower():
        index = 1
    elif "third character" in query.lower():
        index = 2
    # 3. リスト化（すでにfindallでリスト化されている）
    cleaned_list = [normalize(item) for item in matches]
    #cleaned_listに空の要素がある場合、それを削除
    cleaned_list = [c for c in cleaned_list if c]
    # cleaned_listの中身の要素が'r, o, a'といった具合で、','で区切られている場合があるので、それを考慮
    tmp_list = []
    for i in range(len(cleaned_list)):
        if ',' in cleaned_list[i]:
            tmp_list += cleaned_list[i].split(',')
        else:
            tmp_list.append(cleaned_list[i])
    cleaned_list = tmp_list
    cleaned_list = [normalize(c) for c in cleaned_list if normalize(c)]
    # cleaned_listの中身の要素が'r o a'といった具合で、' 'で区切られている場合があるので、それを考慮
    tmp_list = []
    for i in range(len(cleaned_list)):
        if ' ' in cleaned_list[i]:
            tmp_list += cleaned_list[i].split(' ')
        else:
            tmp_list.append(cleaned_list[i])
    cleaned_list = tmp_list
    cleaned_list = [normalize(c) for c in cleaned_list if normalize(c)]
    #cleaned_listが[]だったり、2文字以上の文字がある場合誤検知なのでreturn Noneをする
    if len(cleaned_list) == 0 or any([len(c) > 1 for c in cleaned_list]):
        return None
    if "and end with a letter" in tmp_query:
        matches = re.findall(r"[\"'](.*?)[\"']", query_end)
        cleaned_list_end = [normalize(item) for item in matches]
        if ("between" in query.lower() or ("from" in query.lower() and "to" in query.lower())) and ("between" in query_end.lower() or ("from" in query_end.lower() and "to" in query_end.lower())):
            return 'yes' if (cleaned_list[0] <= normalize(keyward)[index] <= cleaned_list[1]) and (cleaned_list_end[0] <= normalize(keyward)[-1] <= cleaned_list_end[1]) else 'no'
        elif ("between" in query.lower() or ("from" in query.lower() and "to" in query.lower())):
            return 'yes' if (cleaned_list[0] <= normalize(keyward)[index] <= cleaned_list[1]) and (normalize(keyward)[-1] in cleaned_list_end) else 'no'
        elif ("between" in query_end.lower() or ("from" in query_end.lower() and "to" in query_end.lower())): 
            return 'yes' if (normalize(keyward)[index] in cleaned_list) and (cleaned_list_end[0] <= normalize(keyward)[-1] <= cleaned_list_end[1]) else 'no'
        else:
            return 'yes' if (normalize(keyward)[index] in cleaned_list) and (normalize(keyward)[-1] in cleaned_list_end) else 'no'
    elif "between" in query.lower():
        return 'yes' if (cleaned_list[0] <= normalize(keyward)[index] <= cleaned_list[1]) else 'no'
    elif " and is" in query.lower(): #betweenやand end with a letterではないのに、and isが含まれている場合、条件がある場合があるので、それを考慮。例えば and is a type of coastline?とか来る場合、決定論的な答えが出せないので、Noneを返す
        return None
    else:
        return 'yes' if normalize(keyward)[index] in cleaned_list else 'no'

  #keyword end withの場合
  if "keyword end with" in query.lower():
    # 2. 'で囲まれている領域または"で囲まれている領域を取得
    matches = re.findall(r"[\"'](.*?)[\"']", query)
    
    # 3. リスト化（すでにfindallでリスト化されている）
    cleaned_list = [normalize(item) for item in matches]
    cleaned_list = [c for c in cleaned_list if c]
    #cleaned_listが空だったり、5文字以上の文字がある場合誤検知なのでreturn Noneをする
    if len(cleaned_list) == 0 or any([len(c) > 4 for c in cleaned_list]):
        return None
    if "between" in query.lower() or ("from" in query.lower() and "to" in query.lower()):
        return 'yes' if cleaned_list[0] <= normalize(keyward)[-1] <= cleaned_list[1] else 'no'
    else:
        return 'yes' if normalize(keyward)[-1] in cleaned_list else 'no'
  #想定質問2 one of the following または potential keywords
  if ("word one of the following" in query.lower()) or ("potential keywords" in query.lower()) or ("the keyword one of these" in query.lower()):
    # 2. "one of the following"の後ろで一番近い'?'または':'を特定し、それ以前の文字を削除（最短一致）
    match = re.search(r'(word one of the following.*?[?:])|(potential keywords.*?[?:])|(the keyword one of these.*?[?:])', query.lower())
    if match:
        start_index = match.end()
        query = query[start_index:].strip()
    
    # 3. ','で区切ってリスト化
    item_list = query.split(',')
    
    # 4. 各リストの文字にa~z、A~Z、半角スペース以外の文字が入っているならそれは削除
    cleaned_list = [normalize(item) for item in item_list]
    cleaned_list = [c for c in cleaned_list if c]
    #cleaned_listが空だった場合は誤検知なのでreturn Noneをする
    if len(cleaned_list) == 0:
        return None
    #compare_wordsでkeywardとcleaned_listの中の要素を比較して、1つでもTrueがあればyesを返す
    return 'yes' if any([compare_words(keyward, c) for c in cleaned_list]) else 'no'
  #想定質問3 "letter"という文字と"inseide"または"contain"
  if "letter" in query.lower() and ("inside" in query.lower() or "contain" in query.lower()):
      
    # 2. 'で囲まれている領域または"で囲まれている領域を取得
    matches = re.findall(r"[\"'](.*?)[\"']", query)
    if "second word" in query.lower():
        keyward = keyward.split(" ")
        if len(keyward) >= 2:
            keyward = keyward[1]
        else:
            keyward = keyward[0] # 1文字しかない場合はそのまま
    
    # 3. リスト化（すでにfindallでリスト化されている）
    cleaned_list = [normalize(item) for item in matches]
    cleaned_list = [c for c in cleaned_list if c]
    if "between" in query.lower():
        #cleaned_list[0]~cleaned_list[1]のリストを作成
        cleaned_list = [chr(i) for i in range(ord(cleaned_list[0]), ord(cleaned_list[1])+1)]
    #cleaned_listが空だったり、2文字以上の文字がある場合誤検知なのでreturn Noneをする
    if len(cleaned_list) == 0 or any([len(c) > 1 for c in cleaned_list]):
        return None
    if "any" in query.lower():
        #cleaned_listに""が含まれている場合は削除
        cleaned_list = [c for c in cleaned_list if c]
        return 'yes' if any([c in normalize(keyward) for c in cleaned_list]) else 'no'
    else: #bothやallの場合
        return 'yes' if all([c in normalize(keyward) for c in cleaned_list]) else 'no'
  # 想定質問4 Agent aplpha
  match = re.search(r"keyword.*(?:come before|precede) \"([^\"]+)\" .+ order\?$", query.lower())
  if match:
      testword = match.group(1)
      if testword is not None:
          return 'yes' if keyward.lower() < testword.lower() else 'no'
      else:
          return 'no'
  #想定質問4 に同様
  if "lexicographically smaller than" in query.lower() or "alphabetically smaller than" in query.lower():
      
    # 2. 'で囲まれている領域または"で囲まれている領域を取得
    matches = re.findall(r"[\"'](.*?)[\"']", query)
    
    # 3. リスト化（すでにfindallでリスト化されている）
    cleaned_list = [normalize(item) for item in matches]
    cleaned_list = [c for c in cleaned_list if c]
    #cleaned_listが空の場合誤検知なのでreturn Noneをする
    if len(cleaned_list) == 0:
        return None
    return 'yes' if keyward.lower() < cleaned_list[0] else 'no'
  # 想定質問5 キーワードが特定のカテゴリやエージェントに該当するかどうかを確認する質問
  exp = re.compile('are you playing 20 questions|is it a thing|is it agent llama|is it agent|llama3|is it agent meta')
  if re.search(exp, query.lower()):
      return 'yes'
  # 想定質問5と同様
  exp = re.compile('is it a place|is it a landmark|is it a city|is it a person|is it agent gemma|is it agent phi3')
  if re.search(exp, query.lower()):
      return 'no'
  #Does the keyword include the word
  if "does the keyword include the word" in query.lower():
    # does the keyword include the wordより前を削除
    word = query.lower().split("does the keyword include the word")[1]
    word = normalize(word)
    return 'yes' if word in normalize(keyward) else 'no'
  if "does the keyword include the letter" in query.lower():
    # does the keyword include the letterより前を削除
    word = query.lower().split("does the keyword include the letter")[1]
    word = normalize(word)
    if len(word) == 1:
        return 'yes' if word in normalize(keyward) else 'no'
    else:
        return None
  return None


VERSION = 9

# The list of all expected keywords in the game. It has to be updated before submitting your agent for the evaluation
# phase. Apply your own judgement. The longer the list the better coverage, but slower convergence. The coverage is
# arguably more important than convergence, so having many extra keywords is better than having missing words.
#allwords 
# DETERMINE IF SUBMIT OR COMMIT
import os
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
VERBOSE = False
if not os.path.exists(KAGGLE_AGENT_PATH + "weights_VAGOsolutions"):
    KAGGLE_AGENT_PATH = "/app/"
    VERBOSE = True
import pickle

with open(KAGGLE_AGENT_PATH+'large_words.pickle', 'rb') as f:
    allwords = pickle.load(f)
with open(KAGGLE_AGENT_PATH+'large_words_frequency.pickle', 'rb') as f:
    allwords_num = pickle.load(f)
with open(KAGGLE_AGENT_PATH+'english_nouns.pkl', 'rb') as f:
    big_list = pickle.load(f)
with open(KAGGLE_AGENT_PATH+'english_nouns_num.pkl', 'rb') as f:
    big_list_num = pickle.load(f)
#################
# LOAD MODEL INTO MEMORY
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    f"{KAGGLE_AGENT_PATH}weights_VAGOsolutions/",
    torch_dtype = torch.float16,
    device_map = "auto",
    trust_remote_code = True,
)
tokenizer = AutoTokenizer.from_pretrained(f"{KAGGLE_AGENT_PATH}weights_VAGOsolutions/")

##############
# BINARY SEARCH AS QUESTIONER
import random
###############
# LLM MODEL AS ANSWERER

def get_yes_no(question,keyword):
    
    prompt = f'''We are playing a 20 Questions game. In this game, the respondent thinks of a keyword in advance, and the questioner asks up to 20 yes-or-no questions to guess the keyword the respondent is thinking of. You are the respondent, and the keyword you are thinking of is "{keyword}". To deduce the keyword, the questioner will ask questions about the attributes of the keyword or its letter structure. Please answer appropriately with "yes" or "no." Also, for questions about the letter structure, you may refer to the following information about the keyword you are thinking of:

* "{keyword}" is made up of the letters {[char for char in keyword]} arranged in that specific order.
* The first letter of "{keyword}" is "{keyword[0]}".
* The last letter of "{keyword}" is "{keyword[-1]}".

Feel free to use this information as needed.
     
Question from the questioner:

{question}'''
    
    messages = [
        {"role": "system", "content": "Answer yes or no to the following question and nothing else."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask = model_inputs.attention_mask,
        pad_token_id = pad_token_id,
        max_new_tokens=3
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if not "yes" in response.lower(): response = "no"
    else: response = "yes"
        
        
    return response


def answered_yes(obs, question):
    """ Check whether the question has been asked and the answer was 'yes'. """
    try:
        ix = obs.questions.index(question)
        return obs.answers[ix] == 'yes'
    except (ValueError, IndexError):
        return False

def count_letter_frequency(sorted_keywords, position):
  """
  ソート済みキーワードリストのちょうど半分を分割するインデックス番号を探し、
  そのインデックスに対応する文字の出現頻度をカウントします。

  Args:
    sorted_keywords: 指定された位置の文字でソートされたキーワードリスト
    position: 文字の位置 (0: 先頭, 1: 2文字目, -1: 末尾, -2: 最後から2番目)

  Returns:
    tuple: (分割文字, 出現頻度)
  """

  midpoint = len(sorted_keywords) // 2
  split_letter = sorted_keywords[midpoint][position]
  count = 0
  for i,word in enumerate(sorted_keywords):
    try:
        letter = word[position]
        if letter == split_letter:
            count += 1
    except IndexError:
        pass

  return split_letter, count
def find_lowest_frequency_index(results):
  """
  resultsリストから最も出現頻度が低い文字のインデックス番号を返します。

  Args:
    results: (分割文字, 出現頻度) のタプルのリスト

  Returns:
    int: 最も出現頻度が低い文字のインデックス番号
  """

  lowest_frequency = float('inf')  # 出現頻度の最小値を無限大で初期化
  lowest_index = -1  # インデックス番号を初期化

  for index, (_, count) in enumerate(results):
    if count < lowest_frequency:
      lowest_frequency = count
      lowest_index = index

  return lowest_index

class AgentAlpha:
    # This is our search space, it gets modified, so it needs a
    # reinit when testing with a series of games in a row
    keywords = allwords
    nums = allwords_num
    slice_index = None
    is_refinement_by_keywords = False
    refinement_keywords = []
    lower_bound = "aaaa"
    upper_bound = "zzzz"
    exclusion_words = []
    confirmed_guess = False
    keyword_exit = False
    changed_data = False
    chage_data_flag = False
    exit_check_turn = False
    @staticmethod
    def play(obs, cfg):
        if obs.turnType == 'ask':
            AgentAlpha.slice_index = find_closest_half(len(AgentAlpha.keywords))
            if len(AgentAlpha.keywords) > 800:
                test_word = AgentAlpha.keywords[AgentAlpha.slice_index]
                response = f'Does the keyword (in lowercase) precede "{test_word}" in alphabetical order?'
                AgentAlpha.is_refinement_by_keywords = False
            else:
                if AgentAlpha.chage_data_flag:
                    #big_listに差し替え
                    AgentAlpha.keywords = big_list
                    AgentAlpha.nums = big_list_num
                    #lower_boundとupper_boundで初期化。
                    lower_index, upper_index = find_bounds(AgentAlpha.keywords, AgentAlpha.lower_bound, AgentAlpha.upper_bound)
                    AgentAlpha.keywords = AgentAlpha.keywords[lower_index:upper_index+1]
                    AgentAlpha.nums = AgentAlpha.nums[lower_index:upper_index+1]
                    remove_indexs = find_matching_indices(AgentAlpha.keywords, AgentAlpha.exclusion_words)
                    for remove_index in reversed(remove_indexs):
                        del AgentAlpha.keywords[remove_index]
                        del AgentAlpha.nums[remove_index]
                    AgentAlpha.slice_index = find_closest_half(len(AgentAlpha.keywords))
                    AgentAlpha.changed_data = True
                    AgentAlpha.chage_data_flag = False
                    
                adds = ""
                for word in AgentAlpha.keywords:
                    adds += f"{word}, "
                if len(AgentAlpha.keywords) == 0:
                    test_word = random.choice(allwords)
                    prompt = f"Is the keyword one of the following? {test_word}"
                else:
                    prompt = f"Is the keyword one of the following? {adds[:-2]}"
                if len(prompt) < 750:
                    if AgentAlpha.changed_data == False and AgentAlpha.keyword_exit == False:
                        AgentAlpha.exit_check_turn = True
                        AgentAlpha.is_refinement_by_keywords = True
                        AgentAlpha.refinement_keywords = AgentAlpha.keywords
                        return prompt
                if AgentAlpha.changed_data == True:
                    sorted_pairs = sorted(zip(AgentAlpha.nums, AgentAlpha.keywords), reverse=True)
                    result = [string for _, string in sorted_pairs[:AgentAlpha.slice_index]]
                    adds = ""
                    for word in result:
                        adds += f"{word}, "
                    prompt = f"Is the keyword one of the following? {adds[:-2]}"
                    if len(prompt) > 750:
                        test_word = AgentAlpha.keywords[AgentAlpha.slice_index]
                        response = f'Does the keyword (in lowercase) precede "{test_word}" in alphabetical order?'
                        AgentAlpha.is_refinement_by_keywords = False
                    else:
                        response = prompt
                        AgentAlpha.is_refinement_by_keywords = True
                        AgentAlpha.refinement_keywords = result
                else:
                    if AgentAlpha.keyword_exit:
                        sorted_pairs = sorted(zip(AgentAlpha.nums, AgentAlpha.keywords), reverse=True)
                        result = [string for _, string in sorted_pairs[:AgentAlpha.slice_index]]
                        adds = ""
                        for word in result:
                            adds += f"{word}, "
                        prompt = f"Is the keyword one of the following? {adds[:-2]}"
                        if len(prompt) > 750:
                            test_word = AgentAlpha.keywords[AgentAlpha.slice_index]
                            response = f'Does the keyword (in lowercase) precede "{test_word}" in alphabetical order?'
                            AgentAlpha.is_refinement_by_keywords = False
                        else:
                            response = prompt

                            AgentAlpha.is_refinement_by_keywords = True
                            AgentAlpha.refinement_keywords = result
                    else:
                        test_word = AgentAlpha.keywords[AgentAlpha.slice_index]
                        response = f'Does the keyword (in lowercase) precede "{test_word}" in alphabetical order?'
                        AgentAlpha.is_refinement_by_keywords = False


        elif obs.turnType == 'answer':
            try:
              response = simple_question(obs.questions[-1],obs.keyword)
            except:
              response = None

        elif obs.turnType == 'guess':
            if AgentAlpha.is_refinement_by_keywords == False:
                if AgentAlpha.slice_index:
                    if obs.answers[-1] == 'yes':
                        AgentAlpha.upper_bound = AgentAlpha.keywords[AgentAlpha.slice_index]
                        AgentAlpha.keywords = AgentAlpha.keywords[:AgentAlpha.slice_index]
                        AgentAlpha.nums = AgentAlpha.nums[:AgentAlpha.slice_index]
                    else: # no
                        AgentAlpha.lower_bound = AgentAlpha.keywords[AgentAlpha.slice_index]
                        AgentAlpha.keywords = AgentAlpha.keywords[AgentAlpha.slice_index:]
                        AgentAlpha.nums = AgentAlpha.nums[AgentAlpha.slice_index:]
                if  AgentAlpha.keywords:
                    #responseに使うキーワードは、AgentAlpha.numsが一番大きいindexを取得し、そのindexに対応するAgentAlpha.keywordsを取得
                    if AgentAlpha.confirmed_guess:
                        response = AgentAlpha.confirmed_guess
                        AgentAlpha.exclusion_words.append(response)
                        AgentAlpha.confirmed_guess = False
                        #存在確認
                        if response in AgentAlpha.keywords:
                            index = AgentAlpha.keywords.index(response)
                            del AgentAlpha.keywords[index]
                            del AgentAlpha.nums[index]
                    else:
                        response = AgentAlpha.keywords[AgentAlpha.nums.index(max(AgentAlpha.nums))]
                        AgentAlpha.exclusion_words.append(response)
                        #index番号AgentAlpha.nums.index(max(AgentAlpha.nums))を削除
                        del AgentAlpha.keywords[AgentAlpha.nums.index(max(AgentAlpha.nums))]
                        del AgentAlpha.nums[AgentAlpha.nums.index(max(AgentAlpha.nums))]
                else:
                    response = random.choice(allwords)
                    AgentAlpha.exclusion_words.append(response)
            else:
                if obs.answers[-1] == 'yes':
                    keep_strings = AgentAlpha.refinement_keywords
                    kept_numbers = [AgentAlpha.nums[AgentAlpha.keywords.index(keep_string)] for keep_string in keep_strings]
                    AgentAlpha.nums = kept_numbers
                    AgentAlpha.keywords = keep_strings
                    if AgentAlpha.exit_check_turn:
                        AgentAlpha.keyword_exit = True
                        AgentAlpha.exit_check_turn = False
                else: # no
                    # 削除したい文字列に対応する数字を削除する
                    if AgentAlpha.exit_check_turn:
                        AgentAlpha.chage_data_flag = True
                        AgentAlpha.exit_check_turn = False
                    remove_strings = AgentAlpha.refinement_keywords
                    AgentAlpha.exclusion_words += remove_strings
                    for remove_string in remove_strings:
                        index = AgentAlpha.keywords.index(remove_string)
                        del AgentAlpha.keywords[index]
                        del AgentAlpha.nums[index]
                if AgentAlpha.keywords:
                    response = AgentAlpha.keywords[AgentAlpha.nums.index(max(AgentAlpha.nums))]
                    AgentAlpha.exclusion_words.append(response)
                    #index番号AgentAlpha.nums.index(max(AgentAlpha.nums))を削除
                    del AgentAlpha.keywords[AgentAlpha.nums.index(max(AgentAlpha.nums))]
                    del AgentAlpha.nums[AgentAlpha.nums.index(max(AgentAlpha.nums))]
                else:
                    response = random.choice(allwords)
        return response


def agent_fn(obs, cfg):
    """ The main hook, keep it named 'agent_fn' and as the last function in your submission. """

    try:
        # We always try AgentAlpha first, it will either make a meaningful play
        # or return None if it cannot or should not play
        response = AgentAlpha.play(obs, cfg)
        if response is None:
            # アンサーの場合はNoneの場合がある
            response = get_yes_no(obs.questions[-1], obs.keyword)

    except Exception:
        import traceback
        traceback.print_exc()
        if obs.turnType == 'ask':
            test_word = random.choice(allwords)
            response = f'Does the keyword (in lowercase) precede "{test_word}" in alphabetical order?'
        elif obs.turnType == 'answer':
            response = 'no'
        elif obs.turnType == 'guess':
            response = random.choice(allwords)
    return response
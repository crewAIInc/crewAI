#!/usr/bin/env python
import sys
from dbquery.crew import DbQueryCrew

def run():
    inputs = {'user_question': 'Find 10 users from Padova?'}
    result = DbQueryCrew().crew().kickoff(inputs=inputs)
    
if __name__ == "__main__":
    run()
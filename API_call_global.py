# -*- coding: utf-8 -*-
"""
API request to cloud-hosted model

Created on Tue Nov  2 17:09:37 2021

@author: rcpc4
"""

import requests
import json

# Enter hosted URL
url = "https://customerspendapidemo.herokuapp.com/prediction"

# Enter test record
payload = json.dumps({"year_birth":1980,
                        "income":31859,
                        "kidhome":1,
                        "teenhome":0,
                        "recency":3,
                        "numdealspurchases":1,
                        "numwebpurchases":1,
                        "numcatalogpurchases":0,
                        "numstorepurchases":3,
                        "numwebvisitsmonth":7,
                        "enrolment_lag":781})

headers = {'Content-Type':'application/json'}

# Make API request
response = requests.request("POST",url,headers=headers,data=payload)

print(response.text)
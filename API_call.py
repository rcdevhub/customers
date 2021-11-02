# -*- coding: utf-8 -*-
"""
API request to model

Created on Tue Nov  2 17:09:37 2021

@author: rcpc4
"""

import requests
import json

# Enter hosted URL
url = "http://127.0.0.1:4000/prediction"

# Enter test record
payload = json.dumps({"year_birth":1955,
                        "income":80395,
                        "kidhome":0,
                        "teenhome":0,
                        "recency":62,
                        "numdealspurchases":1,
                        "numwebpurchases":6,
                        "numcatalogpurchases":5,
                        "numstorepurchases":12,
                        "numwebvisitsmonth":2,
                        "enrolment_lag":685})

headers = {'Content-Type':'application/json'}

# Make API request
response = requests.request("POST",url,headers=headers,data=payload)

print(response.text)
![Nl2SqlLibrary Archiecture](solid_g-logo-2.png)

# Your One-Stop Workshop for Natural Language to SQL Experimentation

NL2SQL Studio is a comprehensive, open-source toolkit designed to empower developers, data scientists, and analysts in exploring Natural Language to SQL (NL2SQL) and creating production grade NL2SQL pipelines.  Whether you're a seasoned practitioner or just starting out, NL2SQL Studio provides an intuitive interface and a robust set of features to streamline your NL2SQL development workflow.

## Setting up NL2SQL Studio

### Setup

1. **Cloning the repository**

``` 
git clone https://github.com/slatawa/nl2sql.git
```

2. **NL2SQL Modules**

| Sl. No | Module | Description |
|---|---|---|
| 1 | NL2SQL Library | Library code for generating SQL using Vertex AI |
| 2 | User Interface | User interface developed using Streamlit |
| 3 | RAG/PostgreSQL Interface | Interface to add sample questions and generate vector embeddings - required in case you want RAG feature. |

3. **Deployment Order**

| Sl. No | Module | Why |
|---|---|---|
| 1 | NL2SQL Library | URL of the APIs exposed by this module/service is required to be mentioned in the .env file while deploying the UI |
| 1 | RAG/PostgreSQL Interface | URL of the APIs exposed by this module/service is required to be mentioned in the .env file while deploying the UI |
| 2 | User Interface | Config.ini and .env files need to be updated with the API endpoints and Google Client IDs & secrets |



## Executing the NL2SQL Library - Backend


**Prerequisite files**

* app.yaml
* app.py (this will contain the APIs exposed by the backend)
* Wrapper files
* Poetry.lock
* pyproject.toml

Open a new terminal

Navigate to **nl2sql_library** folder

To validate and ensure library dependencies are resolved, create a new Python env, activate the virtual env.  Before running the below steps navigate to nl2sql_library folder in your local

```
python3 -m venv myenv

source ./myenv/bin/activate
```

**Let’s install the dependencies**

```
pip install poetry

poetry install
```

Poetry will install the library dependencies based on the poetry.lock file that is present in the folder.

**Verify the installation by executing the samples**
[ Sample Executors are in the /sample_executors folder ]

Navigate to the sample_executors folder and try any of the following

```
poetry run python linear_executor.py

poetry run python cot_executor.py

poetry run python rag_executor.py
```
**Alternatively, you can run the wrapper class in the main folder like so**

```
poetry run python nl2sql_lib_executors.py linear

poetry run python nl2sql_lib_executors.py cot

poetry run python nl2sql_lib_executors.py rag
```

‘**linear**’, ‘**cot**’, ‘**rag**’ are the type of executors for Linear, Chain of Thought and RAG executors respectively.  Default executor type is ‘linear’ if no executor type is specified as arguments

### Launching the NL2SQL Library service locally

1. Starting the backend as a local service

    * On the terminal window, navigate to **nl2sql_library** folder
    * Type the below command

    ```
    poetry run gunicorn --workers 1 --threads 8 --timeout 0 app:app
    ``` 

    This will start the local service and starts listening to API requests in the http url given as output. Sample output below

    < image to be updated here>

    Note the URL mentioned in the Listening at line - underlined URL.  This will be the URL and port number where the service will be listening for local API requests

2. Testing the local service

    * Open a new terminal window
    * Navigate to **nl2sql_lbrary** folder
    * Open **test.py** and change the URL variable to local host address noted above.  It will be like **http://127.0.0.1:8000".  Sometimes the port number may be different which will be shown as output of above command
    * Run the test.py file

    ```
    python test.py
    ```

    This will call the APIs for generating the SQL using Linear, Chain-of-Thought, RAG executors.  In one pass, it generates only the SQLs and in second pass it generates and executes the SQL as well.  Output will be as shown below

    < image to be updated here>

3. You can also test the service from the python interpreter as well


### Deploying the NL2SQL Library Service on App Engine

1. Creating requirements.txt from Poetry installation

    Launching this setup on App Engine requires multiple commands in the entry point and ATM  this causes problems launching the app. As a workaround we create requirements.txt file from the Poetry virtual env and specify one command for launching the app.  To create requirements.txt, 

    Go back to **nl2sql_library** folder and run below command

    ```
    poetry export --without-hashes --format=requirements.txt > requirements.txt
    ```

    This file is important while deploying on App Engine as the libraries are downloaded and installed on App Engine env.  Any issues with installing the libraries during App engine deployment will cause the app to not launch.


2. Verify library dependencies from requirements.txt

    * Create a new virtual environment
    * Activate the new environment
    * install the dependencies from the requirements.txt file using the following command

    ```
    pip install -r requirements.txt
    ```

    * Validate the environment by executing the sample executors as mentioned above

3. Deploy on App Engine

    To deploy on App Engine, ensure you have app.yaml in the root folder of the service which you want to deploy.  Modify the service name in app.yaml in case required.  Other parameters in the file need not be changed

    Optional steps (Steps 1,2 & 3 below are optional if you have already performed this)

    1. Install **gcloud** on your system from  https://cloud.google.com/sdk/docs/install
    
    2. Authenticate using gcloud using

        ```
        gcloud auth login
        ```
    
        and follow the prompts to authenticate

    3. Set the project

        ```
        gcloud config set project <your project id>
        ```

    4. Use below command to increase timeout on app engine (optional, but run this command if the deployment timeout for default setting) 

        ```
        gcloud config set app/cloud_build_timeout 2000
        ```

        2000 indicates the number of seconds allowed for deployment.  This can be increased to ensure library installation completes during App engine deployment.

    5. If you have the python library packages installed in the same folder (services root folder - for ex: nl2sql_library)

        * Create a new file named **.gitignore**
        * mention the folder name in that file

        This is similar to .gitignore and does not upload the python library packages to App Engine.  Required libraries are installed during deployment based on the requirements.txt file. Else the deployment time might be too high and timeout.

    6. From the services folder which is being deployed here in this case it will be the **nl2sql_library** folder in your local , deploy the service using below command 

        ```
        gcloud app deploy
        ```

        **Note** : If this is the first service on App Engine, the service should be named ‘default’ in app.yaml file. You can only use a custom name if you have a default service deployed.

        Output of the command will be as shown

        < image to be incuded here>

        **Note**- if you see Uploading files are 1000+ there is probably something wrong with your .gitignore file. Please stop deployment and review above step 5

Once deployed, the service will be available on App Engine Services list as shown below

    < image to be included here >

After deployment, on App Engine Services screen, click on the Service.  This will open a new browser tab and perform a GET operation on the default route. If you see a response on browser window as shown below, the deployment is successful

    < image to be included here >


## Setting up NL2SQL UI

Prerequisite files:
* app.yaml
* config.ini
* .env
* nl2sqlstudio_ui.py
* utils.py
* sample_questions.csv

### Setting up UI locally

1. Create a new virtual enironment and activate the same

2. Install dependent libraries.  On terminal, navigate to **UI** folder and run

    ```
    pip install -r requirements.txt
    ```
3. To launch the UI locally, type

    ```
    streamlit run nl2sqlstudio_ui.py
    ```

    Streamlit output will be as shown below

    < image to be included here >

4. Streamlit will initialize and start the server on **http://127.0.0.1:8501"

5. Open the above URL on a browser for UI


### Deploying the UI on App Engine

To deploy on App Engine, ensure you have app.yaml in the root folder of the service which you want to deploy.  Modify the service name in app.yaml in case required.  Other parameters in the file need not be changed
  
Optional steps (Steps 1,2 & 3 below are optional if you have already performed this)

1. Install **gcloud** on your system from  https://cloud.google.com/sdk/docs/install
    
2. Authenticate using gcloud using

        ```
        gcloud auth login
        ```
    
    and follow the prompts to authenticate

3. Set the project

        ```
        gcloud config set project <your project id>
        ```

4. Update **.env** file with the URLs noted for the above 2 services.  Do not change the keys.

        ADD_QUESTION - PG service URL
        EXECUTORS - NL2SQL Library service URL

5. Update **config.ini** (you will need to update this if you are enabling SSO authentication)

6. Update sample_questions.txt (sample questions related to your dataset)

7. Goto folder where the app.yaml file is available

8. Update “config.ini” file with the Client ID and Client secret from the Credentials screen of APIs and Services

9. Update “.env’ file with the URL link of the NL2SQL Library back-end service deployed on App Engine as shown above and replace value for EXECUTORS 

10. Execute the following code

    ```
    gcloud app dwploy
    ```

If the deployment is successful, App Engine endpoint will be provided in the output.

This will be the URL for the default service.  If you have deployed with a different service name, the URL will be prefixed with the service name.  For example:

**nl2sqlstudio-ui-dot-sl-test-project-363109.uc.r.appspot.com**

< images to be included here>

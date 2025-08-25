# Setting Up Your Open-Source Endpoint

> NOTE: If you do not wish to purchase $50 of compute credits for T2 (for using dedicated endpoints) you can instead skip the following set-up and simply use the serverless endpoints offered at: 

- `openai/gpt-oss-20b`

## Your Generator

First, you'll want to navigate to [api.together.ai/models](https://api.together.ai/models), and search for the model we'll be using today: 

- `gpt-oss`

We're going to select the OpenAI GPT-OSS 20B model by clicking on it:

![image](./images/Z82ArVL.png)

Next, we're going to click on "Create Dedicated Endpoint" to spin up a dedicated endpoint. 

![image](./images/dWqtZ6i.png)

You'll want to set your settings as follows and then click "Deploy": 

![image](./images/eZvZGZo%20-%20Imgur.png)

> NOTE: Please ensure you have an Auto-shutdown selected - a value like `1 hour` is useful to ensure your endpoint does not spin down during class.

After you click "Deploy" - you should see the endpoint spinning up, as well as a name for your new endpoint!

> NOTE: You'll want to make sure you get an API key from together.ai as well! You can follow the instructions [here](https://docs.together.ai/reference/authentication-1)

## Your Embeddings 

Together offers serverless endpoints for embedding models, we'll be using the [BAAI-BGE-Large-1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model today!

- `BAAI/bge-large-en-v1.5`

### ❓ Question #1: 

What is the difference between serverless and dedicated endpoints?

### ✅ Answer #1: 
#### **Serverless Endpoints**  

* No setup required: Can use them immediately without any configuration

* No cost for setup: Free to use (pay only for actual usage)

* Shared resources: Multiple users share the same infrastructure

* Automatic scaling: The platform handles scaling based on demand

* No management overhead: No need to worry about starting/stopping endpoints

* In my codebase: openai/gpt-oss-20b (the default serverless option)

#### **Dedicated Endpoints**

* Requires setup: Need to manually create and configure them
* Setup cost: Requires purchasing compute credits (e.g., $50 for T2)
* Dedicated resources: Allows exclusive access to the infrastructure
* Manual management: MUST control when to start/stop the endpoint
* Auto-shutdown feature: Must set automatic shutdown timers (e.g., 1 hour) to control costs
* Better performance: More consistent latency and throughput since you're not sharing resources
* Custom configuration: You can optimize settings for your specific use case

#### TIMER COSTS - SUT THEM DOWN!!!
* Timer starts immediately when you deploy, not when you start using it
* No usage = still charging: Even if you're not making API calls, you're still being charged for the dedicated resources

**How to Set Automatic Shutdown**
When Creating a Dedicated Endpoint:
* Go to api.together.ai/models
* Search for gpt-oss (my model) and click on it
* Click "Create Dedicated Endpoint"
* In the deployment settings, look for "Auto-shutdown" or "Shutdown timer"
* Set it to a reasonable time like 1 hour or 2 hours
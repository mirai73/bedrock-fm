# bedrock_fm

**Convenience classes to interact with Bedrock Foundation Models**

Amazon Bedrock provides a unified API to invoke foundation models in the form of `bedrock.invoke_model()` and `bedrock.invoke_model_with_response_stream()`. These unified methods require a `modelId` and a stringified JSON `body` containing the model specific input parameters. Additional parameters, such as `accept`, and `contentType` can also be specified.

This simplicity comes at the cost that developers need to know the model specific format of the `body` payload. Moreover, the payloads being completely different, does not allow to easy swapping out a model for another. Another disadvantage is that the generic `body` cannot be type annotated and the developer cannot therefore take advantage of IDE autocompletion features.

## The `bedrock_fm` library

The `bedrock_fm` library exposes a separate class for each of the Bedrock model family, each exposing a consistent `generate()` API which is common across all models. The same method can be used to get a stream instead of a full completion by passing the `stream=True` as parameter. To obtain a detailed response including the original prompt, the body passed to the `invoke_*` method and timing information you can pass the parameter `details=True`. The API is fully typed, including the different return types based on the `stream` and `details` parameters values.

The output generation can be tuned with the optional `temperature`, `top_p` and `stop_words` parameters which can be passed at the instance creation time (in the class constructor) and overridden at generation time in the `generate` method.

All models create a `boto3` client at the time of instantiation using the default session. To customize the client creation, for example to access Bedrock in a different region or account one can:

- use [environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables) (such as `AWS_PROFILE` and `AWS_DEFAULT_REGION`)
- call `boto3.setup_default_session()` method before the foundation model instances are created
- create a `boto3.Session` and pass it to the FM model constructor via `session=`

Model specific parameters other than **temperature**, **top P** and **stop words**, can be provided via the `extra_args` parameters as a Dict, both in the constructors and in the `generate` method call. By specifying the `extra_args` in the constructor of the foundation model class makes it easier to swap out a FM for another without changing the business logic.

Models supporting a chat modality (eg Llama2 Chat and Claude models) can also be invoked with the `chat()` API. This API accepts an ordered array of conversation items, consisting of `System`, `Human` and `Assistant`.

## Installation

## Build the wheel

This project uses [poetry](https://python-poetry.org/). Follow their [instructions to install](https://python-poetry.org/docs/#installation).

Clone the repo and build the wheel via:

```
poetry build
```

This will create a wheel in `./dist/` folder that you can then install in your own project via `pip` or `poetry`

## Generation examples

**Basic use**

```py
from bedrock_fm import from_model_id, Model

fm = from_model_id(Model.AMAZON_TITAN_TEXT_EXPRESS_V1)
fm.generate("Hi. Tell me a joke?")
```

**Instantiate a model of a specific provider**

This is equivalent to the above, but also validates that the `model_id` is compatible with the model provider. You can use the model ID string and are not obliged to use the provided `Model` enumeration.

```py
from bedrock_fm import Titan

fm = Titan.from_id("amazon.titan-text-express-v1")
fm.generate("Hi. Tell me a joke?")

# This fails
fm = Titan.from_id("anthropic.claude-v1")
```

**Streaming**

```py
from bedrock_fm import from_model_id, Model

fm = from_model_id(Model.ANTHROPIC_CLAUDE_INSTANT_V1)
for t in fm.generate("Hi. Tell me a joke?", stream=True):
    print(t)
```

**Bedrock client customization via default session**

```py
from bedrock_fm import from_model_id
import boto3

boto3.setup_default_session(region_name='us-east-1')
fm = from_model_id("amazon.titan-text-express-v1")
```

**Bedrock client customization via boto3.Session**

```py
from bedrock_fm import from_model_id
import boto3

session = boto3.Session(region_name='us-east-1')
fm = from_model_id("amazon.titan-text-express-v1", session=session)
```

**Common Foundation Model parameters**

```py
from bedrock_fm import from_model_id

# You can setup parameters at the model instance level
fm = from_model_id("anthropic.claude-instant-v1", temperature=0.5, top_p=1)
# You can override parameters value when invoking the generation functions - also for stream
print(fm.generate("Hi. Tell me a joke?", token_count=100)[0])

```

**Model specific parameters**

```py
from bedrock_fm import from_model_id

# Set up extra parameter for the model instance
fm = from_model_id("anthropic.claude-instant-v1", extra_args={'top_k': 200})
for t in fm.generate("Hi. Tell me a joke?", stream=True):
    print(t)

# Override the instance parameters

for t in fm.generate("Hi. Tell me a joke?", stream=True, extra_args={'top_k': 400}):
    print(t)
```

**Get inference details**

```py
from bedrock_fm import from_model_id

fm = from_model_id("anthropic.claude-instant-v1")

print(fm.generate("Tell me a joke?", details=True))

""""
CompletionDetails(output=['Sorry - this model is designed to avoid potentially inappropriate content. Please see our content limitations page for more information.'], response={'inputTextTokenCount': 4, 'results': [{'tokenCount': 31, 'outputText': 'Sorry - this model is designed to avoid potentially inappropriate content. Please see our content limitations page for more information.', 'completionReason': 'CONTENT_FILTERED'}]}, prompt='Tell me a joke', body='{"inputText": "Tell me a joke", "textGenerationConfig": {"maxTokenCount": 500, "stopSequences": [], "temperature": 0.7, "topP": 1}}', latency=1.531674861907959)
"""
```

## Chat

When using the `chat()` API, we need to provide an ordered conversation array. If you use a `System` prompt, it must be the first element and cannot repeat.
You normally alternate `Human` and `Assistant` elements, finishing with a `Human` element.

```py

from bedrock_fm import Llama2Chat

conversation = [System("You are an helpful travel agent"), Human("What is the capital of France")]

fm = Llama2Chat.from_id("meta.llama2-13b-chat-v1")

answer = fm.chat(conversation)[0]
print(answer)

# Append the answer to the current conversation
conversation.append(Assistant(answer))

# Ask a new question based on the conversation
conversation.append(Human("Tell me more about this city"))

answer = fm.chat(conversation)[0]
print(answer)
```

Try removing the System prompt and see how the answers change.

### 🚀🚀 NEW! Claude 3 and multi-modal chat 🚀🚀

Anthropic has introduced a new Message API which maps nicely to the chat model. You can still use `generate`, but you can better leverage Claude 3 capabilities by using the `chat` API.

Here is an example of a multimodal chat:

```py
from bedrock_fm import Claude3, Human, Assistant, System
from PIL import Image
import boto3

session = boto3.Session(region_name="us-east-1")

fm = Claude3.from_id("anthropic.claude-3-sonnet-20240229-v1:0", session = session)

resp = fm.chat([System("You are an expert art dealer"), Human(content="Tell me about this painting", images=[Image.open("monet.png")])])

print(resp[0])
```

## Embeddings

Embedding API provides a `generate` method that generates document embeddings by default. It also provide a specific `generate_for_documents` and `generate_for_query` methods.

**Titan embeddings**

```py

from bedrock_fm import from_model_id

emb = from_model_id("amazon.titan-embed-text-v1")
print(emb.generate(["Tell me a joke"])[0])
```

**Cohere embeddings**

```py

from bedrock_fm import from_model_id

emb = from_model_id("cohere.embed-english-v3")
print(emb.generate_for_documents(["Paris is in France", "Rome is in Italy", "Paris is also called Ville Lumiere"]))
print(emb.generate_for_query("Where is Paris?"))
```

## Image generation

This library supports image generation with StableDiffusion and Titan models. Check the `image.ipynb` notebook for some examples.

## Throttling

To cope with throttling exceptions you can use libraries like [backoff](https://pypi.org/project/backoff/)

```python
import backoff

@backoff.on_exception(backoff.expo, br.exceptions.ThrottlingException)
def generate(prompt):
    return fm.generate(prompt)

generate("Hello how are you?")

```

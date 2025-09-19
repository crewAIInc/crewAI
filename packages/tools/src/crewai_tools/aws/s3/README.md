# AWS S3 Tools

## Description

These tools provide a way to interact with Amazon S3, a cloud storage service.

## Installation

Install the crewai_tools package

```shell
pip install 'crewai[tools]'
```

## AWS Connectivity

The tools use `boto3` to connect to AWS S3.
You can configure your environment to use AWS IAM roles, see [AWS IAM Roles documentation](https://docs.aws.amazon.com/sdk-for-python/v1/developer-guide/iam-roles.html#creating-an-iam-role)

Set the following environment variables:

- `CREW_AWS_REGION`
- `CREW_AWS_ACCESS_KEY_ID`
- `CREW_AWS_SEC_ACCESS_KEY`

## Usage

To use the AWS S3 tools in your CrewAI agents, import the necessary tools and include them in your agent's configuration:

```python
from crewai_tools.aws.s3 import S3ReaderTool, S3WriterTool

# For reading from S3
@agent
def file_retriever(self) -> Agent:
    return Agent(
        config=self.agents_config['file_retriever'],
        verbose=True,
        tools=[S3ReaderTool()]
    )

# For writing to S3
@agent
def file_uploader(self) -> Agent:
    return Agent(
        config=self.agents_config['file_uploader'],
        verbose=True,
        tools=[S3WriterTool()]
    )
```

These tools can be used to read from and write to S3 buckets within your CrewAI workflows. Make sure you have properly configured your AWS credentials as mentioned in the AWS Connectivity section above.

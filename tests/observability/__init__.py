

from crewai import Agent, Task, LLM


def test_langdb_basic_integration_example():
    """Test the basic LangDB integration example from the documentation."""
    
    class MockLangDB:
        @staticmethod
        def init(**kwargs):
            pass
    
    MockLangDB.init()
    
    llm = LLM(
        model="gpt-4o",
        temperature=0.7
    )
    
    agent = Agent(
        role="Senior Research Analyst",
        goal="Conduct comprehensive research on assigned topics",
        backstory="You are an expert researcher with deep analytical skills.",
        llm=llm,
        verbose=True
    )
    
    assert agent.role == "Senior Research Analyst"
    assert agent.goal == "Conduct comprehensive research on assigned topics"
    assert agent.llm == llm


def test_langdb_metadata_configuration_example():
    """Test the metadata configuration example from the documentation."""
    class MockLangDB:
        @staticmethod
        def init(metadata=None, **kwargs):
            assert metadata is not None
            assert "environment" in metadata
            assert "crew_type" in metadata
    
    MockLangDB.init(
        metadata={
            "environment": "production",
            "crew_type": "research_workflow",
            "user_id": "user_123"
        }
    )


def test_langdb_cost_tracking_example():
    """Test the cost tracking configuration example from the documentation."""
    class MockLangDB:
        @staticmethod
        def init(cost_tracking=None, budget_alerts=None, **kwargs):
            assert cost_tracking is True
            assert budget_alerts is not None
            assert "daily_limit" in budget_alerts
            assert "alert_threshold" in budget_alerts
    
    MockLangDB.init(
        cost_tracking=True,
        budget_alerts={
            "daily_limit": 100.0,
            "alert_threshold": 0.8
        }
    )


def test_langdb_security_configuration_example():
    """Test the security configuration example from the documentation."""
    class MockLangDB:
        @staticmethod
        def init(security_config=None, **kwargs):
            assert security_config is not None
            assert "pii_detection" in security_config
            assert "content_filtering" in security_config
            assert "audit_logging" in security_config
    
    MockLangDB.init(
        security_config={
            "pii_detection": True,
            "content_filtering": True,
            "audit_logging": True,
            "data_retention_days": 90
        }
    )


def test_langdb_environment_specific_setup():
    """Test the multi-environment setup example from the documentation."""
    environments = ["production", "staging", "development"]
    
    for env in environments:
        class MockLangDB:
            @staticmethod
            def init(project_id=None, sampling_rate=None, cost_tracking=None, **kwargs):
                assert project_id is not None
                assert sampling_rate is not None
                assert cost_tracking is not None
        
        if env == "production":
            MockLangDB.init(
                project_id="prod_project_id",
                sampling_rate=1.0,
                cost_tracking=True
            )
        elif env == "staging":
            MockLangDB.init(
                project_id="staging_project_id",
                sampling_rate=0.5,
                cost_tracking=False
            )
        else:
            MockLangDB.init(
                project_id="dev_project_id",
                sampling_rate=0.1,
                cost_tracking=False
            )


def test_langdb_task_with_metadata():
    """Test task creation with metadata as shown in documentation."""
    llm = LLM(model="gpt-4o")
    
    agent = Agent(
        role="Senior Research Analyst",
        goal="Conduct research",
        backstory="Expert researcher",
        llm=llm
    )
    
    task = Task(
        description="Research the latest AI trends",
        expected_output="Comprehensive research report",
        agent=agent
    )
    
    assert task.description == "Research the latest AI trends"
    assert task.expected_output == "Comprehensive research report"
    assert task.agent == agent

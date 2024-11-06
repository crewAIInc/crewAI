"""Test Knowledge creation and querying functionality."""

from pathlib import Path

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


def test_single_short_string():
    # Create a knowledge base with a single short string
    content = "Brandon's favorite color is blue and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)
    knowledge_base = Knowledge(sources=[string_source])

    # Perform a query
    query = "What is Brandon's favorite color?"
    results = knowledge_base.query(query)

    # Assert that the results contain the expected information
    assert any("blue" in result.lower() for result in results)


def test_single_2k_character_string():
    # Create a 2k character string with various facts about Brandon
    content = (
        "Brandon is a software engineer who lives in San Francisco. "
        "He enjoys hiking and often visits the trails in the Bay Area. "
        "Brandon has a pet dog named Max, who is a golden retriever. "
        "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
        "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
        "He is also a fan of Mexican cuisine, especially tacos and burritos. "
        "Brandon plays the guitar and often performs at local open mic nights. "
        "He is learning French and plans to visit Paris next year. "
        "Brandon is passionate about technology and often attends tech meetups in the city. "
        "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
        "Brandon's favorite color is blue, and he often wears blue shirts. "
        "He enjoys cooking and often tries new recipes on weekends. "
        "Brandon is a morning person and likes to start his day with a run in the park. "
        "He is also a coffee enthusiast and enjoys trying different coffee blends. "
        "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
        "He is also a fan of board games and often hosts game nights at his place. "
        "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
        "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
        "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
        "He is also a fan of the Golden State Warriors and enjoys watching their games. "
    )
    string_source = StringKnowledgeSource(content=content)
    knowledge_base = Knowledge(sources=[string_source])

    # Perform a query
    query = "What is Brandon's favorite movie?"
    results = knowledge_base.query(query)

    # Assert that the results contain the expected information
    assert any("inception" in result.lower() for result in results)


def test_multiple_short_strings():
    # Create multiple short string sources
    contents = [
        "Brandon loves hiking.",
        "Brandon has a dog named Max.",
        "Brandon enjoys painting landscapes.",
    ]
    string_sources = [StringKnowledgeSource(content=content) for content in contents]
    knowledge_base = Knowledge(sources=string_sources)

    # Perform a query
    query = "What is the name of Brandon's pet?"
    results = knowledge_base.query(query)

    # Assert that the correct information is retrieved
    assert any("max" in result.lower() for result in results)


def test_multiple_2k_character_strings():
    # Create multiple 2k character strings with various facts about Brandon
    contents = [
        (
            "Brandon is a software engineer who lives in San Francisco. "
            "He enjoys hiking and often visits the trails in the Bay Area. "
            "Brandon has a pet dog named Max, who is a golden retriever. "
            "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
            "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
            "He is also a fan of Mexican cuisine, especially tacos and burritos. "
            "Brandon plays the guitar and often performs at local open mic nights. "
            "He is learning French and plans to visit Paris next year. "
            "Brandon is passionate about technology and often attends tech meetups in the city. "
            "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
            "Brandon's favorite color is blue, and he often wears blue shirts. "
            "He enjoys cooking and often tries new recipes on weekends. "
            "Brandon is a morning person and likes to start his day with a run in the park. "
            "He is also a coffee enthusiast and enjoys trying different coffee blends. "
            "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
            "He is also a fan of board games and often hosts game nights at his place. "
            "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
            "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
            "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
            "He is also a fan of the Golden State Warriors and enjoys watching their games. "
        )
        * 2,  # Repeat to ensure it's 2k characters
        (
            "Brandon loves traveling and has visited over 20 countries. "
            "He is fluent in Spanish and often practices with his friends. "
            "Brandon's favorite city is Barcelona, where he enjoys the architecture and culture. "
            "He is a foodie and loves trying new cuisines, with a particular fondness for sushi. "
            "Brandon is an avid cyclist and participates in local cycling events. "
            "He is also a photographer and enjoys capturing landscapes and cityscapes. "
            "Brandon is a tech enthusiast and follows the latest trends in gadgets and software. "
            "He is also a fan of virtual reality and owns a VR headset. "
            "Brandon's favorite book is 'The Hitchhiker's Guide to the Galaxy'. "
            "He enjoys watching documentaries and learning about history and science. "
            "Brandon is a coffee lover and has a collection of coffee mugs from different countries. "
            "He is also a fan of jazz music and often attends live performances. "
            "Brandon is a member of a local running club and participates in marathons. "
            "He is also a volunteer at a local animal shelter and helps with dog walking. "
            "Brandon's favorite holiday is Christmas, and he enjoys decorating his home. "
            "He is also a fan of classic movies and has a collection of DVDs. "
            "Brandon is a mentor for young professionals and enjoys giving career advice. "
            "He is also a fan of puzzles and enjoys solving them in his free time. "
            "Brandon's favorite sport is soccer, and he often plays with his friends. "
            "He is also a fan of FC Barcelona and enjoys watching their matches. "
        )
        * 2,  # Repeat to ensure it's 2k characters
    ]
    string_sources = [StringKnowledgeSource(content=content) for content in contents]
    knowledge_base = Knowledge(sources=string_sources)

    # Perform a query
    query = "What is Brandon's favorite book?"
    results = knowledge_base.query(query)

    # Assert that the correct information is retrieved
    assert any(
        "the hitchhiker's guide to the galaxy" in result.lower() for result in results
    )


def test_single_short_file(tmpdir):
    # Create a single short text file
    content = "Brandon's favorite sport is basketball."
    file_path = Path(tmpdir.join("short_file.txt"))
    with open(file_path, "w") as f:
        f.write(content)

    file_source = TextFileKnowledgeSource(file_path=file_path)
    knowledge_base = Knowledge(sources=[file_source])

    # Perform a query
    query = "What sport does Brandon like?"
    results = knowledge_base.query(query)

    # Assert that the results contain the expected information
    assert any("basketball" in result.lower() for result in results)


def test_single_2k_character_file(tmpdir):
    # Create a single 2k character text file with various facts about Brandon
    content = (
        "Brandon is a software engineer who lives in San Francisco. "
        "He enjoys hiking and often visits the trails in the Bay Area. "
        "Brandon has a pet dog named Max, who is a golden retriever. "
        "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
        "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
        "He is also a fan of Mexican cuisine, especially tacos and burritos. "
        "Brandon plays the guitar and often performs at local open mic nights. "
        "He is learning French and plans to visit Paris next year. "
        "Brandon is passionate about technology and often attends tech meetups in the city. "
        "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
        "Brandon's favorite color is blue, and he often wears blue shirts. "
        "He enjoys cooking and often tries new recipes on weekends. "
        "Brandon is a morning person and likes to start his day with a run in the park. "
        "He is also a coffee enthusiast and enjoys trying different coffee blends. "
        "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
        "He is also a fan of board games and often hosts game nights at his place. "
        "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
        "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
        "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
        "He is also a fan of the Golden State Warriors and enjoys watching their games. "
    ) * 2  # Repeat to ensure it's 2k characters
    file_path = Path(tmpdir.join("long_file.txt"))
    with open(file_path, "w") as f:
        f.write(content)

    file_source = TextFileKnowledgeSource(file_path=file_path)
    knowledge_base = Knowledge(sources=[file_source])

    # Perform a query
    query = "What is Brandon's favorite movie?"
    results = knowledge_base.query(query)

    # Assert that the results contain the expected information
    assert any("inception" in result.lower() for result in results)


def test_multiple_short_files(tmpdir):
    # Create multiple short text files
    contents = [
        "Brandon lives in New York.",
        "Brandon works as a software engineer.",
        "Brandon enjoys cooking Italian food.",
    ]
    file_paths = []
    for i, content in enumerate(contents):
        file_path = Path(tmpdir.join(f"file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)

    file_sources = [TextFileKnowledgeSource(file_path=path) for path in file_paths]
    knowledge_base = Knowledge(sources=file_sources)

    # Perform a query
    query = "Where does Brandon live?"
    results = knowledge_base.query(query)

    # Assert that the correct information is retrieved
    assert any("new york" in result.lower() for result in results)


def test_multiple_2k_character_files(tmpdir):
    # Create multiple 2k character text files with various facts about Brandon
    contents = [
        (
            "Brandon loves traveling and has visited over 20 countries. "
            "He is fluent in Spanish and often practices with his friends. "
            "Brandon's favorite city is Barcelona, where he enjoys the architecture and culture. "
            "He is a foodie and loves trying new cuisines, with a particular fondness for sushi. "
            "Brandon is an avid cyclist and participates in local cycling events. "
            "He is also a photographer and enjoys capturing landscapes and cityscapes. "
            "Brandon is a tech enthusiast and follows the latest trends in gadgets and software. "
            "He is also a fan of virtual reality and owns a VR headset. "
            "Brandon's favorite book is 'The Hitchhiker's Guide to the Galaxy'. "
            "He enjoys watching documentaries and learning about history and science. "
            "Brandon is a coffee lover and has a collection of coffee mugs from different countries. "
            "He is also a fan of jazz music and often attends live performances. "
            "Brandon is a member of a local running club and participates in marathons. "
            "He is also a volunteer at a local animal shelter and helps with dog walking. "
            "Brandon's favorite holiday is Christmas, and he enjoys decorating his home. "
            "He is also a fan of classic movies and has a collection of DVDs. "
            "Brandon is a mentor for young professionals and enjoys giving career advice. "
            "He is also a fan of puzzles and enjoys solving them in his free time. "
            "Brandon's favorite sport is soccer, and he often plays with his friends. "
            "He is also a fan of FC Barcelona and enjoys watching their matches. "
        )
        * 2,  # Repeat to ensure it's 2k characters
        (
            "Brandon is a software engineer who lives in San Francisco. "
            "He enjoys hiking and often visits the trails in the Bay Area. "
            "Brandon has a pet dog named Max, who is a golden retriever. "
            "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
            "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
            "He is also a fan of Mexican cuisine, especially tacos and burritos. "
            "Brandon plays the guitar and often performs at local open mic nights. "
            "He is learning French and plans to visit Paris next year. "
            "Brandon is passionate about technology and often attends tech meetups in the city. "
            "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
            "Brandon's favorite color is blue, and he often wears blue shirts. "
            "He enjoys cooking and often tries new recipes on weekends. "
            "Brandon is a morning person and likes to start his day with a run in the park. "
            "He is also a coffee enthusiast and enjoys trying different coffee blends. "
            "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
            "He is also a fan of board games and often hosts game nights at his place. "
            "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
            "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
            "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
            "He is also a fan of the Golden State Warriors and enjoys watching their games. "
        )
        * 2,  # Repeat to ensure it's 2k characters
    ]
    file_paths = []
    for i, content in enumerate(contents):
        file_path = Path(tmpdir.join(f"long_file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)

    file_sources = [TextFileKnowledgeSource(file_path=path) for path in file_paths]
    knowledge_base = Knowledge(sources=file_sources)

    # Perform a query
    query = "What is Brandon's favorite book?"
    results = knowledge_base.query(query)

    # Assert that the correct information is retrieved
    assert any(
        "the hitchhiker's guide to the galaxy" in result.lower() for result in results
    )


def test_hybrid_string_and_files(tmpdir):
    # Create string sources
    string_contents = [
        "Brandon is learning French.",
        "Brandon visited Paris last summer.",
    ]
    string_sources = [
        StringKnowledgeSource(content=content) for content in string_contents
    ]

    # Create file sources
    file_contents = [
        "Brandon prefers tea over coffee.",
        "Brandon's favorite book is 'The Alchemist'.",
    ]
    file_paths = []
    for i, content in enumerate(file_contents):
        file_path = Path(tmpdir.join(f"file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)

    file_sources = [TextFileKnowledgeSource(file_path=path) for path in file_paths]

    # Combine string and file sources
    knowledge_base = Knowledge(sources=string_sources + file_sources)

    # Perform a query
    query = "What is Brandon's favorite book?"
    results = knowledge_base.query(query)

    # Assert that the correct information is retrieved
    assert any("the alchemist" in result.lower() for result in results)


def test_pdf_knowledge_source():
    # Get the directory of the current file
    current_dir = Path(__file__).parent
    # Construct the path to the PDF file
    pdf_path = current_dir / "crewai_quickstart.pdf"

    # Create a PDFKnowledgeSource
    pdf_source = PDFKnowledgeSource(file_path=pdf_path)
    knowledge_base = Knowledge(sources=[pdf_source])

    # Perform a query
    query = "How do you create a crew?"
    results = knowledge_base.query(query)

    print("Results from querying PDFKnowledgeSource:", results)
    # Assert that the correct information is retrieved
    assert any(
        "crewai create crew latest-ai-development" in result.lower()
        for result in results
    )

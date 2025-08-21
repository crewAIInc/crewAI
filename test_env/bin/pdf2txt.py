#!/Users/jonathanhill/src/crewai-professional/crewai_upstream/test_env/bin/python3.12
"""A command line tool for extracting text and images from PDF and
output it to plain text, html, xml or tags.
"""

import argparse
import logging
import sys
from typing import Any, Container, Iterable, List, Optional

import pdfminer.high_level
from pdfminer.layout import LAParams
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.utils import AnyIO

logging.basicConfig()

OUTPUT_TYPES = ((".htm", "html"), (".html", "html"), (".xml", "xml"), (".tag", "tag"))


def float_or_disabled(x: str) -> Optional[float]:
    if x.lower().strip() == "disabled":
        return None
    try:
        return float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {x}")


def extract_text(
    files: Iterable[str] = [],
    outfile: str = "-",
    laparams: Optional[LAParams] = None,
    output_type: str = "text",
    codec: str = "utf-8",
    strip_control: bool = False,
    maxpages: int = 0,
    page_numbers: Optional[Container[int]] = None,
    password: str = "",
    scale: float = 1.0,
    rotation: int = 0,
    layoutmode: str = "normal",
    output_dir: Optional[str] = None,
    debug: bool = False,
    disable_caching: bool = False,
    **kwargs: Any,
) -> AnyIO:
    if not files:
        raise PDFValueError("Must provide files to work upon!")

    if output_type == "text" and outfile != "-":
        for override, alttype in OUTPUT_TYPES:
            if outfile.endswith(override):
                output_type = alttype

    if outfile == "-":
        outfp: AnyIO = sys.stdout
        if sys.stdout.encoding is not None:
            codec = "utf-8"
    else:
        outfp = open(outfile, "wb")

    for fname in files:
        with open(fname, "rb") as fp:
            pdfminer.high_level.extract_text_to_fp(fp, **locals())
    return outfp


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "files",
        type=str,
        default=None,
        nargs="+",
        help="One or more paths to PDF files.",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"pdfminer.six v{pdfminer.__version__}",
    )
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Use debug logging level.",
    )
    parser.add_argument(
        "--disable-caching",
        "-C",
        default=False,
        action="store_true",
        help="If caching or resources, such as fonts, should be disabled.",
    )

    parse_params = parser.add_argument_group(
        "Parser",
        description="Used during PDF parsing",
    )
    parse_params.add_argument(
        "--page-numbers",
        type=int,
        default=None,
        nargs="+",
        help="A space-seperated list of page numbers to parse.",
    )
    parse_params.add_argument(
        "--pagenos",
        "-p",
        type=str,
        help="A comma-separated list of page numbers to parse. "
        "Included for legacy applications, use --page-numbers "
        "for more idiomatic argument entry.",
    )
    parse_params.add_argument(
        "--maxpages",
        "-m",
        type=int,
        default=0,
        help="The maximum number of pages to parse.",
    )
    parse_params.add_argument(
        "--password",
        "-P",
        type=str,
        default="",
        help="The password to use for decrypting PDF file.",
    )
    parse_params.add_argument(
        "--rotation",
        "-R",
        default=0,
        type=int,
        help="The number of degrees to rotate the PDF "
        "before other types of processing.",
    )

    la_params = LAParams()  # will be used for defaults
    la_param_group = parser.add_argument_group(
        "Layout analysis",
        description="Used during layout analysis.",
    )
    la_param_group.add_argument(
        "--no-laparams",
        "-n",
        default=False,
        action="store_true",
        help="If layout analysis parameters should be ignored.",
    )
    la_param_group.add_argument(
        "--detect-vertical",
        "-V",
        default=la_params.detect_vertical,
        action="store_true",
        help="If vertical text should be considered during layout analysis",
    )
    la_param_group.add_argument(
        "--line-overlap",
        type=float,
        default=la_params.line_overlap,
        help="If two characters have more overlap than this they "
        "are considered to be on the same line. The overlap is specified "
        "relative to the minimum height of both characters.",
    )
    la_param_group.add_argument(
        "--char-margin",
        "-M",
        type=float,
        default=la_params.char_margin,
        help="If two characters are closer together than this margin they "
        "are considered to be part of the same line. The margin is "
        "specified relative to the width of the character.",
    )
    la_param_group.add_argument(
        "--word-margin",
        "-W",
        type=float,
        default=la_params.word_margin,
        help="If two characters on the same line are further apart than this "
        "margin then they are considered to be two separate words, and "
        "an intermediate space will be added for readability. The margin "
        "is specified relative to the width of the character.",
    )
    la_param_group.add_argument(
        "--line-margin",
        "-L",
        type=float,
        default=la_params.line_margin,
        help="If two lines are close together they are considered to "
        "be part of the same paragraph. The margin is specified "
        "relative to the height of a line.",
    )
    la_param_group.add_argument(
        "--boxes-flow",
        "-F",
        type=float_or_disabled,
        default=la_params.boxes_flow,
        help="Specifies how much a horizontal and vertical position of a "
        "text matters when determining the order of lines. The value "
        "should be within the range of -1.0 (only horizontal position "
        "matters) to +1.0 (only vertical position matters). You can also "
        "pass `disabled` to disable advanced layout analysis, and "
        "instead return text based on the position of the bottom left "
        "corner of the text box.",
    )
    la_param_group.add_argument(
        "--all-texts",
        "-A",
        default=la_params.all_texts,
        action="store_true",
        help="If layout analysis should be performed on text in figures.",
    )

    output_params = parser.add_argument_group(
        "Output",
        description="Used during output generation.",
    )
    output_params.add_argument(
        "--outfile",
        "-o",
        type=str,
        default="-",
        help="Path to file where output is written. "
        'Or "-" (default) to write to stdout.',
    )
    output_params.add_argument(
        "--output_type",
        "-t",
        type=str,
        default="text",
        help="Type of output to generate {text,html,xml,tag}.",
    )
    output_params.add_argument(
        "--codec",
        "-c",
        type=str,
        default="utf-8",
        help="Text encoding to use in output file.",
    )
    output_params.add_argument(
        "--output-dir",
        "-O",
        default=None,
        help="The output directory to put extracted images in. If not given, "
        "images are not extracted.",
    )
    output_params.add_argument(
        "--layoutmode",
        "-Y",
        default="normal",
        type=str,
        help="Type of layout to use when generating html "
        "{normal,exact,loose}. If normal,each line is"
        " positioned separately in the html. If exact"
        ", each character is positioned separately in"
        " the html. If loose, same result as normal "
        "but with an additional newline after each "
        "text line. Only used when output_type is html.",
    )
    output_params.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="The amount of zoom to use when generating html file. "
        "Only used when output_type is html.",
    )
    output_params.add_argument(
        "--strip-control",
        "-S",
        default=False,
        action="store_true",
        help="Remove control statement from text. "
        "Only used when output_type is xml.",
    )

    return parser


def parse_args(args: Optional[List[str]]) -> argparse.Namespace:
    parsed_args = create_parser().parse_args(args=args)

    # Propagate parsed layout parameters to LAParams object
    if parsed_args.no_laparams:
        parsed_args.laparams = None
    else:
        parsed_args.laparams = LAParams(
            line_overlap=parsed_args.line_overlap,
            char_margin=parsed_args.char_margin,
            line_margin=parsed_args.line_margin,
            word_margin=parsed_args.word_margin,
            boxes_flow=parsed_args.boxes_flow,
            detect_vertical=parsed_args.detect_vertical,
            all_texts=parsed_args.all_texts,
        )

    if parsed_args.page_numbers:
        parsed_args.page_numbers = {x - 1 for x in parsed_args.page_numbers}

    if parsed_args.pagenos:
        parsed_args.page_numbers = {int(x) - 1 for x in parsed_args.pagenos.split(",")}

    if parsed_args.output_type == "text" and parsed_args.outfile != "-":
        for override, alttype in OUTPUT_TYPES:
            if parsed_args.outfile.endswith(override):
                parsed_args.output_type = alttype

    return parsed_args


def main(args: Optional[List[str]] = None) -> int:
    parsed_args = parse_args(args)
    outfp = extract_text(**vars(parsed_args))
    outfp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

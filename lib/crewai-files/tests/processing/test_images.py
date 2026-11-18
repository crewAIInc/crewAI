"""Tests for open_image, the format-restricted Pillow open."""

import io
import struct
from unittest.mock import patch

from PIL import Image, PsdImagePlugin, features
from crewai_files import FileBytes, ImageFile, PDFFile
from crewai_files.processing.constraints import (
    ANTHROPIC_CONSTRAINTS,
    AZURE_CONSTRAINTS,
    ImageConstraints,
    ProviderConstraints,
)
from crewai_files.processing.exceptions import (
    UnsupportedFileTypeError,
    UnsupportedImageFormatError,
)
from crewai_files.processing.images import SUPPORTED_IMAGE_FORMATS, open_image
from crewai_files.processing.processor import FileProcessor
from crewai_files.processing.transformers import (
    get_image_dimensions,
    optimize_image,
    resize_image,
)
from crewai_files.processing.validators import validate_image
import pytest


def _psd(width: int = 10_000, height: int = 10_000) -> bytes:
    """A minimal 8-bit grayscale PSD header that Pillow's PSD parser accepts."""
    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, 1, height, width, 8, 1)
    return header + struct.pack(">IIIH", 0, 0, 0, 0)


def _encode(fmt: str, size: tuple[int, int] = (16, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=fmt)
    return buffer.getvalue()


def _image_file(data: bytes, filename: str = "photo.png") -> ImageFile:
    return ImageFile(source=FileBytes(data=data, filename=filename))


class TestOpenImage:
    @pytest.mark.parametrize("fmt", SUPPORTED_IMAGE_FORMATS)
    def test_opens_supported_formats(self, fmt: str) -> None:
        if fmt == "WEBP" and not features.check("webp"):
            pytest.skip("Pillow built without WEBP support")
        with open_image(_encode(fmt)) as img:
            assert img.format == fmt
            assert img.size == (16, 16)

    def test_psd_is_detected_by_pillow_without_the_restriction(self) -> None:
        # Guards the fixture: if Pillow stopped recognising it, the rejection
        # tests below would pass without proving anything.
        with Image.open(io.BytesIO(_psd(64, 64))) as img:
            assert img.format == "PSD"

    def test_rejects_psd_without_running_its_parser(self) -> None:
        with (
            patch.object(
                PsdImagePlugin.PsdImageFile,
                "_open",
                side_effect=AssertionError("PSD parser ran"),
            ),
            pytest.raises(UnsupportedImageFormatError) as exc_info,
        ):
            with open_image(_psd(), filename="photo.png", content_type="image/png"):
                pass

        assert str(exc_info.value) == (
            "Cannot process image 'photo.png': it is not a valid PNG, JPEG, GIF "
            "or WEBP image"
        )
        assert exc_info.value.file_name == "photo.png"
        assert exc_info.value.content_type == "image/png"

    @pytest.mark.skipif(
        not features.check("jpg_2000"), reason="Pillow built without OpenJPEG"
    )
    def test_rejects_jpeg_2000(self) -> None:
        with pytest.raises(UnsupportedImageFormatError):
            with open_image(_encode("JPEG2000")):
                pass

    def test_rejects_bytes_that_are_not_an_image(self) -> None:
        with pytest.raises(UnsupportedImageFormatError):
            with open_image(b"not an image"):
                pass


class TestEveryPillowOpenIsRestricted:
    """A PSD named and declared as a PNG is refused wherever Pillow would open it."""

    def test_validate_image_raises_even_without_raise_on_error(self) -> None:
        constraints = ImageConstraints(
            max_size_bytes=100 * 1024 * 1024,
            max_width=8000,
            max_height=8000,
            supported_formats=("image/png",),
        )

        with pytest.raises(UnsupportedImageFormatError):
            validate_image(_image_file(_psd()), constraints, raise_on_error=False)

    def test_get_image_dimensions(self) -> None:
        with pytest.raises(UnsupportedImageFormatError):
            get_image_dimensions(_image_file(_psd()))

    def test_resize_image(self) -> None:
        with pytest.raises(UnsupportedImageFormatError):
            resize_image(_image_file(_psd()), 8000, 8000)

    def test_optimize_image(self) -> None:
        with pytest.raises(UnsupportedImageFormatError):
            optimize_image(_image_file(_psd()), target_size_bytes=10)


class TestProcessorAutoMode:
    def test_rejects_psd_for_a_provider_with_dimension_limits(self) -> None:
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)

        with pytest.raises(UnsupportedImageFormatError):
            processor.process(_image_file(_psd()))

    def test_rejects_psd_that_would_be_optimized(self) -> None:
        # No dimension limits, so the first Pillow open is optimize_image's.
        constraints = ProviderConstraints(
            name="test", image=ImageConstraints(max_size_bytes=10)
        )
        processor = FileProcessor(constraints=constraints)

        with pytest.raises(UnsupportedImageFormatError):
            processor.process(_image_file(_psd()))

    def test_still_resizes_a_supported_image(self) -> None:
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)
        file = _image_file(_encode("PNG", size=(9000, 100)))

        result = processor.process(file)

        assert isinstance(result, ImageFile)
        with Image.open(io.BytesIO(result.read())) as img:
            assert img.size[0] <= 8000


class TestProcessFiles:
    def test_skips_only_the_rejected_image(self) -> None:
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)
        good = _image_file(_encode("PNG"), filename="good.png")

        result = processor.process_files(
            {"good": good, "bad": _image_file(_psd(64, 64), filename="bad.png")}
        )

        assert result == {"good": good}

    def test_strict_mode_still_raises_for_a_file_type_the_provider_rejects(
        self,
    ) -> None:
        processor = FileProcessor(constraints=AZURE_CONSTRAINTS)
        pdf = PDFFile(
            source=FileBytes(data=b"%PDF-1.4", filename="report.pdf"), mode="strict"
        )

        with pytest.raises(UnsupportedFileTypeError):
            processor.process_files({"good": _image_file(_encode("PNG")), "pdf": pdf})

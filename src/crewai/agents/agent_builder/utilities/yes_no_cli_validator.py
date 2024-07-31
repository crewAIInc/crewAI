from prompt_toolkit.validation import Validator, ValidationError


class YesNoValidator(Validator):
    def validate(self, document):
        text = document.text.lower()
        if text not in ["y", "n", "yes", "no"]:
            raise ValidationError(message="Please enter Y/N")

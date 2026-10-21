from pydantic import BaseModel, ConfigDict

ADVANCED = {"x-advanced": True}
"""JSON schema extra marking a field that most users can leave untouched."""


class EnergyCostModel(BaseModel):
    """Base class for the library's models.

    The docstring placed directly under a field becomes its ``description`` in the JSON schema,
    and the class docstring becomes the model's ``description``, so keep both to one short sentence.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

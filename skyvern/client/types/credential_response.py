# This file was auto-generated by Fern from our API Definition.

from ..core.pydantic_utilities import UniversalBaseModel
import pydantic
from .credential_response_credential import CredentialResponseCredential
from .credential_type_output import CredentialTypeOutput
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import typing


class CredentialResponse(UniversalBaseModel):
    """
    Response model for credential operations.
    """

    credential_id: str = pydantic.Field()
    """
    Unique identifier for the credential
    """

    credential: CredentialResponseCredential = pydantic.Field()
    """
    The credential data
    """

    credential_type: CredentialTypeOutput = pydantic.Field()
    """
    Type of the credential
    """

    name: str = pydantic.Field()
    """
    Name of the credential
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow

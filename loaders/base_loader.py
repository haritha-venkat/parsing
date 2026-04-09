from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return the file extensions supported by this loader."""
        raise NotImplementedError

    @abstractmethod
    def load(self, file_path: str | Path) -> list[dict]:
        """
        Load a file and return a list of document dicts.

        Args:
            file_path: Absolute or relative path to the target file.

        Returns:
            list[dict]: Extracted documents with keys
                ``source``, ``page``, ``text``, ``metadata``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """

    def _validate(self, file_path: Path) -> None:
        """
        Shared validation: existence check + extension check.

        Args:
            file_path: Path object to validate.

        Raises:
            FileNotFoundError: File does not exist on disk.
            ValueError: Extension not in ``supported_extensions``.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported extension '{file_path.suffix}' for "
                f"{self.__class__.__name__}. "
                f"Expected one of: {self.supported_extensions}"
            )

import html
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class NotesReader(BaseReader):
    """
    Apple Notes reader for macOS Notes app.
    
    Reads notes from the Apple Notes SQLite database.
    """

    def __init__(self, include_folders: bool = True, include_metadata: bool = True) -> None:
        """
        Initialize the Notes reader.
        
        Args:
            include_folders: Whether to include folder information in metadata
            include_metadata: Whether to include creation/modification dates and other metadata
        """
        self.include_folders = include_folders
        self.include_metadata = include_metadata

    def _find_notes_database(self) -> Path | None:
        """
        Find the Apple Notes database file.
        
        Returns:
            Path to the Notes database or None if not found
        """
        # Standard location for Apple Notes database
        notes_db_path = Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NoteStore.sqlite"
        
        if notes_db_path.exists():
            return notes_db_path
        
        # Alternative locations to check
        alternative_paths = [
            Path.home() / "Library" / "Containers" / "com.apple.Notes" / "Data" / "Library" / "Notes" / "NotesV7.storedata",
            Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NotesV1.storedata"
        ]
        
        for path in alternative_paths:
            if path.exists():
                return path
                
        return None

    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean HTML content from notes to extract plain text.
        
        Args:
            html_content: Raw HTML content from notes
            
        Returns:
            Cleaned plain text
        """
        if not html_content:
            return ""
        
        # Decode HTML entities
        text = html.unescape(html_content)
        
        # Remove HTML tags but preserve some structure
        # Replace common block elements with newlines
        text = re.sub(r'</(div|p|br|h[1-6]|li)>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<(div|p|br|h[1-6]|li)[^>]*>', '\n', text, flags=re.IGNORECASE)
        
        # Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = text.strip()
        
        return text

    def _format_timestamp(self, timestamp: float) -> str:
        """
        Convert Apple Notes timestamp to readable format.
        
        Apple Notes uses Core Data timestamps (seconds since Jan 1, 2001)
        
        Args:
            timestamp: Core Data timestamp
            
        Returns:
            Formatted datetime string
        """
        if not timestamp:
            return "Unknown"
        
        # Convert Core Data timestamp to Unix timestamp
        # Core Data epoch is Jan 1, 2001, Unix epoch is Jan 1, 1970
        core_data_epoch = datetime(2001, 1, 1)
        unix_epoch = datetime(1970, 1, 1)
        offset = (core_data_epoch - unix_epoch).total_seconds()
        
        unix_timestamp = timestamp + offset
        
        try:
            dt = datetime.fromtimestamp(unix_timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return "Invalid Date"

    def load_data(self, notes_db_path: str | None = None, **load_kwargs: Any) -> list[Document]:
        """
        Load notes from Apple Notes database.
        
        Args:
            notes_db_path: Path to Notes database (auto-detected if None)
            **load_kwargs:
                max_count (int): Maximum number of notes to read
                folder_filter (str): Only read notes from folders containing this string
                
        Returns:
            List of Document objects containing notes
        """
        max_count = load_kwargs.get("max_count", -1)
        folder_filter = load_kwargs.get("folder_filter", None)
        
        # Find database path
        if notes_db_path:
            db_path = Path(notes_db_path)
        else:
            db_path = self._find_notes_database()
            
        if not db_path or not db_path.exists():
            raise FileNotFoundError(
                "Apple Notes database not found. Make sure you're running on macOS and Notes app has been used."
            )
        
        documents = []
        
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            cursor = conn.cursor()
            
            # Query to get notes with their content and metadata
            # This query works with the typical Apple Notes database schema
            query = """
            SELECT 
                n.Z_PK as note_id,
                n.ZTITLE as title,
                n.ZSNIPPET as snippet,
                n.ZCREATIONDATE as creation_date,
                n.ZMODIFICATIONDATE as modification_date,
                nb.ZDATA as content_data,
                f.ZTITLE as folder_name
            FROM ZICNOTEDATA n
            LEFT JOIN ZICNOTEBODY nb ON n.Z_PK = nb.ZNOTE
            LEFT JOIN ZICFOLDER f ON n.ZFOLDER = f.Z_PK
            WHERE n.ZMARKEDFORDELETION = 0
            """
            
            # Add folder filter if specified
            if folder_filter:
                query += f" AND f.ZTITLE LIKE '%{folder_filter}%'"
            
            query += " ORDER BY n.ZMODIFICATIONDATE DESC"
            
            # Add limit if specified
            if max_count > 0:
                query += f" LIMIT {max_count}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                note_id = row['note_id']
                title = row['title'] or f"Note {note_id}"
                snippet = row['snippet'] or ""
                creation_date = row['creation_date']
                modification_date = row['modification_date']
                content_data = row['content_data']
                folder_name = row['folder_name'] or "Notes"
                
                # Extract content
                content = ""
                if content_data:
                    try:
                        # Apple Notes stores content as binary data, often in a proprietary format
                        # For basic implementation, we'll try to decode as text and clean HTML
                        raw_content = content_data.decode('utf-8', errors='ignore')
                        content = self._clean_html_content(raw_content)
                    except Exception:
                        # Fallback to snippet if content extraction fails
                        content = snippet
                
                # Use snippet if content is empty
                if not content and snippet:
                    content = snippet
                
                # Skip empty notes
                if not content:
                    continue
                
                # Prepare text for document
                note_text = f"Title: {title}\n\n{content}"
                
                # Prepare metadata
                metadata = {
                    "note_id": str(note_id),
                    "title": title,
                    "source": "Apple Notes",
                }
                
                if self.include_folders:
                    metadata["folder"] = folder_name
                
                if self.include_metadata:
                    metadata["creation_date"] = self._format_timestamp(creation_date)
                    metadata["modification_date"] = self._format_timestamp(modification_date)
                
                # Create document
                doc = Document(
                    text=note_text,
                    metadata=metadata,
                    id_=f"note_{note_id}"
                )
                
                documents.append(doc)
            
            conn.close()
            
        except sqlite3.Error as e:
            raise RuntimeError(f"Error reading Apple Notes database: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading notes: {e}")
        
        return documents


def find_notes_database() -> Path | None:
    """
    Utility function to find the Apple Notes database.
    
    Returns:
        Path to the Notes database or None if not found
    """
    reader = NotesReader()
    return reader._find_notes_database()
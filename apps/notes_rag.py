"""
Notes RAG example using the unified interface.
Supports Apple Notes on macOS.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks

from apps.notes_data.LEANN_notes_reader import NotesReader


class NotesRAG(BaseRAGExample):
    """RAG example for Apple Notes processing."""

    def __init__(self):
        # Set default values BEFORE calling super().__init__
        self.max_items_default = -1  # Process all notes by default
        self.embedding_model_default = (
            "sentence-transformers/all-MiniLM-L6-v2"  # Fast 384-dim model
        )

        super().__init__(
            name="Notes",
            description="Process and query Apple Notes with LEANN",
            default_index_name="notes_index",
        )

    def _add_specific_arguments(self, parser):
        """Add notes-specific arguments."""
        notes_group = parser.add_argument_group("Notes Parameters")
        notes_group.add_argument(
            "--notes-db-path",
            type=str,
            default=None,
            help="Path to Apple Notes database (auto-detected if not specified)",
        )
        notes_group.add_argument(
            "--folder-filter",
            type=str,
            default=None,
            help="Only process notes from folders containing this string",
        )
        notes_group.add_argument(
            "--include-folders", 
            action="store_true", 
            default=True,
            help="Include folder information in metadata (default: True)"
        )
        notes_group.add_argument(
            "--chunk-size", 
            type=int, 
            default=512, 
            help="Text chunk size (default: 512)"
        )
        notes_group.add_argument(
            "--chunk-overlap", 
            type=int, 
            default=50, 
            help="Text chunk overlap (default: 50)"
        )

    def _find_notes_database(self) -> Path | None:
        """Auto-detect Apple Notes database."""
        notes_db_path = Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NoteStore.sqlite"
        
        if notes_db_path.exists():
            return notes_db_path
        
        # Alternative locations
        alternative_paths = [
            Path.home() / "Library" / "Containers" / "com.apple.Notes" / "Data" / "Library" / "Notes" / "NotesV7.storedata",
            Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NotesV1.storedata"
        ]
        
        for path in alternative_paths:
            if path.exists():
                return path
                
        return None

    async def load_data(self, args) -> list[str]:
        """Load notes and convert to text chunks."""
        # Determine notes database path
        if args.notes_db_path:
            db_path = Path(args.notes_db_path)
            if not db_path.exists():
                print(f"Specified notes database path does not exist: {args.notes_db_path}")
                return []
        else:
            print("Auto-detecting Apple Notes database...")
            db_path = self._find_notes_database()

        if not db_path:
            print("Apple Notes database not found!")
            print("Make sure you're running on macOS and the Notes app has been used.")
            print("You can also specify the database path manually with --notes-db-path")
            return []

        print(f"Found Notes database: {db_path}")

        # Create reader
        reader = NotesReader(
            include_folders=args.include_folders,
            include_metadata=True
        )

        try:
            # Load notes
            print("Reading notes from database...")
            
            # Prepare load kwargs
            load_kwargs = {}
            if args.max_items > 0:
                load_kwargs["max_count"] = args.max_items
            if args.folder_filter:
                load_kwargs["folder_filter"] = args.folder_filter
            
            documents = reader.load_data(
                notes_db_path=str(db_path),
                **load_kwargs
            )

            if not documents:
                print("No notes found to process!")
                if args.folder_filter:
                    print(f"Try removing the folder filter '{args.folder_filter}' or check folder names")
                return []

            print(f"Successfully loaded {len(documents)} notes")

            # Convert documents to text chunks
            print("Converting notes to text chunks...")
            all_texts = create_text_chunks(
                documents, 
                chunk_size=args.chunk_size, 
                chunk_overlap=args.chunk_overlap
            )

            print(f"Created {len(all_texts)} text chunks from {len(documents)} notes")
            return all_texts

        except FileNotFoundError as e:
            print(f"Database error: {e}")
            return []
        except RuntimeError as e:
            print(f"Error processing notes: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []


if __name__ == "__main__":
    import asyncio

    # Check platform
    if sys.platform != "darwin":
        print("\n⚠️  Warning: This example is designed for macOS (Apple Notes)")
        print("   Windows/Linux support coming soon!\n")

    # Example queries for notes RAG
    print("\n📝 Apple Notes RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'Find my grocery lists'")
    print("- 'What ideas did I write about the project?'")
    print("- 'Show me notes about travel plans'")
    print("- 'Find meeting notes from last week'")
    print("- 'What recipes did I save?'")
    print("\nNote: You may need to grant Full Disk Access to your terminal")
    print("in System Preferences → Privacy & Security → Full Disk Access\n")

    rag = NotesRAG()
    asyncio.run(rag.run())
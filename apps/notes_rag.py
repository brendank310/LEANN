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
        # Platform check with helpful guidance
        import sys
        if sys.platform != "darwin":
            print("\n❌ Apple Notes is only available on macOS")
            print("\n🔄 Alternative options for your platform:")
            if sys.platform.startswith("linux"):
                print("  • Use document_rag.py with text/markdown files")
                print("  • Export notes from other apps (OneNote, Notion) to supported formats")
                print("  • Consider using browser_rag.py for web-based note-taking tools")
            elif sys.platform == "win32":
                print("  • Use document_rag.py with text/markdown files") 
                print("  • Export from OneNote or other Windows note apps")
                print("  • Consider using browser_rag.py for web-based note-taking tools")
            print(f"\n💡 Current platform: {sys.platform}")
            print("📚 See README.md for platform-specific alternatives")
            return []
        
        # Determine notes database path
        if args.notes_db_path:
            db_path = Path(args.notes_db_path)
            if not db_path.exists():
                print(f"❌ Specified notes database path does not exist: {args.notes_db_path}")
                print("💡 Try auto-detection by omitting --notes-db-path")
                return []
        else:
            print("🔍 Auto-detecting Apple Notes database...")
            db_path = self._find_notes_database()

        if not db_path:
            print("❌ Apple Notes database not found!")
            print("\n🔧 Troubleshooting steps:")
            print("  1. Make sure you're running on macOS")
            print("  2. Open the Notes app and create at least one note")
            print("  3. Grant Full Disk Access to your terminal:")
            print("     System Preferences → Privacy & Security → Full Disk Access")
            print("  4. Try specifying the database path manually with --notes-db-path")
            print("\n📍 Common database locations:")
            print("  • ~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite")
            print("  • ~/Library/Containers/com.apple.Notes/Data/Library/Notes/NotesV7.storedata")
            return []

        print(f"✅ Found Notes database: {db_path}")

        # Create reader
        reader = NotesReader(
            include_folders=args.include_folders,
            include_metadata=True
        )

        try:
            # Load notes
            print("📖 Reading notes from database...")
            
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
                print("❌ No notes found to process!")
                if args.folder_filter:
                    print(f"🔍 Try removing the folder filter '{args.folder_filter}' or check folder names")
                    print("💡 Available folders can be seen by running without --folder-filter first")
                else:
                    print("🔧 Troubleshooting:")
                    print("  • Make sure you have notes in the Apple Notes app")
                    print("  • Check Full Disk Access permissions")
                    print("  • Try a different database path with --notes-db-path")
                return []

            print(f"✅ Successfully loaded {len(documents)} notes")

            # Convert documents to text chunks
            print("🔄 Converting notes to text chunks...")
            all_texts = create_text_chunks(
                documents, 
                chunk_size=args.chunk_size, 
                chunk_overlap=args.chunk_overlap
            )

            print(f"✅ Created {len(all_texts)} text chunks from {len(documents)} notes")
            print("🚀 Ready for semantic search!")
            return all_texts

        except FileNotFoundError as e:
            print(f"❌ Database error: {e}")
            print("💡 Make sure the Notes app has been opened and contains notes")
            return []
        except RuntimeError as e:
            print(f"❌ Error processing notes: {e}")
            print("🔧 This might be a permissions issue - check Full Disk Access")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("🆘 Please check your Notes database and permissions")
            return []


if __name__ == "__main__":
    import asyncio

    # Platform-specific startup information
    print("\n📝 Apple Notes RAG Example")
    print("=" * 50)
    
    if sys.platform == "darwin":
        print("✅ Running on macOS - Apple Notes supported!")
        print("\n🔧 Setup Requirements:")
        print("  1. Grant Full Disk Access to your terminal:")
        print("     System Preferences → Privacy & Security → Full Disk Access")
        print("  2. Ensure you have notes in the Apple Notes app")
        
        print("\n🤖 AI Model Options:")
        print("  • Cloud AI (OpenAI): Set OPENAI_API_KEY environment variable")
        print("  • Local AI (Ollama): Install Ollama and pull a model (e.g., llama3.2:1b)")
        print("    - macOS: brew install ollama && ollama pull llama3.2:1b")
        print("    - Use --llm ollama --llm-model llama3.2:1b")
        
        print("\n📊 Example Usage Scenarios:")
        print("  # Quick start with OpenAI (requires API key)")
        print("  python -m apps.notes_rag --query 'Find my grocery lists'")
        print("  ")
        print("  # Local AI with Ollama (fully private)")
        print("  python -m apps.notes_rag --llm ollama --llm-model llama3.2:1b --query 'Find my recipes'")
        print("  ")
        print("  # Search specific folder with custom chunking")
        print("  python -m apps.notes_rag --folder-filter 'Work' --chunk-size 1024 --query 'meeting notes'")
    else:
        print(f"❌ Platform: {sys.platform} - Apple Notes not supported")
        print("\n🔄 Alternative Solutions:")
        if sys.platform.startswith("linux"):
            print("  📁 Document RAG: python -m apps.document_rag --data-dir ~/Documents")
            print("  🌐 Browser RAG: python -m apps.browser_rag --query 'research topics'")
            print("  💾 Export notes from web-based tools (Notion, Google Keep) as text/markdown")
        elif sys.platform == "win32":
            print("  📁 Document RAG: python -m apps.document_rag --data-dir C:\\Users\\YourName\\Documents")
            print("  📝 Export from OneNote or other Windows note apps")
            print("  🌐 Browser RAG: python -m apps.browser_rag --query 'research topics'")
        
        print("\n💡 To continue anyway (will fail gracefully):")
        print("  python -m apps.notes_rag --query 'test query'")
    
    print("\n🎯 Example Queries (if on macOS):")
    print("  - 'Find my grocery lists'")
    print("  - 'What ideas did I write about the project?'")
    print("  - 'Show me notes about travel plans'")
    print("  - 'Find meeting notes from last week'")
    print("  - 'What recipes did I save?'")
    print()

    rag = NotesRAG()
    asyncio.run(rag.run())
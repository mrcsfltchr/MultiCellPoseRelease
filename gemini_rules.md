# Gemini Project Rules: GUVpose

This file lists rules for a project to migrate the GUVpose codebased defined here into a new structure which will make it easier to maintain and extend. The original code structure and definition summary is defined in '@cellpose_context.md'. A draft new MVC structure is defined in '@guv_app_context.md'.

## 1. Architectural Standards
- **Pattern**: Strict Model-View-Controller (MVC) separation.
- **Model**: Logic in `dynamics.py`, `train.py`, and `models.py`. No Qt imports here.
- **View**: UI definitions in `guiparts.py` and `menus.py`.
- **Controller**: Logic to coordinate UI events with core processing.

## 2. Coding Practices
- **Threading**: Use `QtCore.QThread` or gRPC streams for all processing tasks.
- **Imports**: Avoid circular dependencies. Core logic must remain importable without the GUI.
- **Style**: Adhere to PEP 8 and use type hints.

## 3. AI Agent Protocols
- **Memory**: Always consult `@cellpose_context.md` as the primary architectural map of the original code structure. For the 
- **Brevity**: Use bullet points for summaries. Avoid long blocks of uninterrupted text.
- **Timeouts**: If a task requires more than 10 files, request the context in batches.
- **Refactoring**: When refactoring, provide a step-by-step 'Migration Plan' before outputting code.
- **Migration**: Use the definitions and function names in both '@cellpose_context.md' and '@guv_app_context.md' to guide which specific files in the original GUVpose codebase you need to read to understand the original function then plan the corresponding MVC version. Do not load the whole project into context.
- **Function implementation**: when migrate functions do not rewrite function definitions that could otherwise be directly imported from the current codebase, unless these functions are not compatible with the MVC structure.
import pandas as pd
import numpy as np
import logging
import pkgutil
import importlib
import inspect
import os
from typing import List, Dict
from guv_app.plugins.interface import AnalysisPlugin
import guv_app.plugins

_logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Manages and executes analysis plugins.
    """
    def __init__(self):
        self.plugins = {}
        self.discover_plugins()

    def discover_plugins(self):
        """Automatically discovers and registers plugins from the plugins package."""
        for _, name, _ in pkgutil.iter_modules(guv_app.plugins.__path__, guv_app.plugins.__name__ + "."):
            try:
                module = importlib.import_module(name)
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, AnalysisPlugin):
                        if not inspect.isabstract(obj):
                            try:
                                self.register_plugin(obj())
                            except Exception as e:
                                _logger.error(f"Failed to instantiate plugin {obj.__name__} from {name}: {e}")
            except Exception as e:
                _logger.error(f"Failed to import plugin module {name}: {e}")

    def register_plugin(self, plugin: AnalysisPlugin):
        """Registers a new analysis plugin."""
        self.plugins[plugin.name] = plugin
        _logger.info(f"Registered analysis plugin: {plugin.name}")

    def get_available_plugins(self) -> List[str]:
        return list(self.plugins.keys())

    def run_analysis(self, image, masks, classes=None, filename=None, plugins=None, plugin_params=None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Runs specified plugins on the data and returns a dictionary of results.
        """
        results = {}
        
        # Determine list of plugin instances to run
        plugins_list = []
        if plugins is None:
            plugins_list = list(self.plugins.values())
        else:
            for p in plugins:
                if isinstance(p, str):
                    if p in self.plugins:
                        plugins_list.append(self.plugins[p])
                elif isinstance(p, AnalysisPlugin):
                    plugins_list.append(p)

        for plugin in plugins_list:
            name = plugin.name
            
            # 1. Start with defaults from the plugin definition
            params = {}
            definitions = plugin.get_parameter_definitions()
            for param_key, param_def in definitions.items():
                if 'default' in param_def:
                    params[param_key] = param_def['default']

            # 2. Override with any user-provided parameters
            if plugin_params and name in plugin_params:
                params.update(plugin_params[name])

            try:
                df = plugin.run(image, masks, classes=classes, **params)
                if df is not None and not df.empty:
                    if filename:
                        df.insert(0, 'filename', filename)
                    results[name] = df
            except Exception as e:
                _logger.error(f"Error running plugin {name}: {e}")

        return results

    def run_visualization(self, plugin: AnalysisPlugin, image: np.ndarray, masks: np.ndarray, 
                          classes: np.ndarray = None, plugin_params: Dict = None) -> np.ndarray:
        """
        Runs the visualize method of a plugin, handling parameter defaults.
        """
        params = {}
        definitions = plugin.get_parameter_definitions()
        
        # 1. Apply defaults
        for param_key, param_def in definitions.items():
            if 'default' in param_def:
                params[param_key] = param_def['default']

        # 2. Override with user params
        if plugin_params:
            params.update(plugin_params)
            
        # 3. Execute
        return plugin.visualize(image, masks, classes=classes, **params)

    def save_results(self, results: Dict[str, pd.DataFrame], filename: str) -> List[str]:
        """
        Saves analysis results to CSV files.
        """
        saved_files = []
        if not filename:
            return saved_files
        
        base = os.path.splitext(filename)[0]
        
        for plugin_name, df in results.items():
            if df is not None and not df.empty:
                # Sanitize plugin name for filename
                safe_name = "".join(x for x in plugin_name if x.isalnum() or x in "._- ").replace(" ", "_")
                csv_path = f"{base}_{safe_name}.csv"
                try:
                    df.to_csv(csv_path, index=False)
                    saved_files.append(csv_path)
                    _logger.info(f"Saved analysis results to {csv_path}")
                except Exception as e:
                    _logger.error(f"Failed to save CSV {csv_path}: {e}")
        return saved_files
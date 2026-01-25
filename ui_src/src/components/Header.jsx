import React from 'react';
import { Mic, FileText, Sun, Moon } from 'lucide-react';

const Header = ({ darkMode, toggleTheme }) => {
  return (
    <nav className="relative z-50 glass border-b border-white/5 sticky top-0">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          <div className="flex items-center gap-4">
            <a href="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:shadow-indigo-500/40 transition-shadow">
               <Mic className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
                Chatterbox
              </span>
            </a>
          </div>
          <div className="flex items-center gap-6">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors border border-white/5 group"
            >
              {darkMode ? (
                 <Moon className="w-5 h-5 text-indigo-400 theme-icon-moon" />
              ) : (
                 <Sun className="w-5 h-5 text-yellow-400 theme-icon-sun" />
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Header;

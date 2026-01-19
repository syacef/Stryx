import React from 'react';
import { Leaf, Trash2, Globe, PlayCircle } from 'lucide-react';

const FeedCard = ({ feed, isSelected, onSelect, onDelete }) => (
  <div
    onClick={() => onSelect(feed)}
    className={`group relative p-5 rounded-2xl transition-all cursor-pointer border-2 
      ${isSelected 
        ? 'bg-emerald-900/20 border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.1)]' 
        : 'bg-stone-900/40 border-stone-800 hover:border-emerald-700/50'}`}
  >
    <div className="flex justify-between items-start">
      <div className="flex gap-4">
        <div className={`p-3 rounded-xl ${isSelected ? 'bg-emerald-500 text-white' : 'bg-stone-800 text-emerald-500'}`}>
          {feed.source === 'direct' ? <Globe size={20} /> : <Leaf size={20} />}
        </div>
        <div>
          <h3 className="font-bold text-stone-100">{feed.name}</h3>
          <p className="text-xs font-mono text-stone-500 mt-1 truncate max-w-[200px]">
            {feed.rtspUrl}
          </p>
        </div>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(feed.id); }}
        className="opacity-0 group-hover:opacity-100 p-2 hover:bg-red-500/10 text-stone-500 hover:text-red-400 transition-all rounded-lg"
      >
        <Trash2 size={18} />
      </button>
    </div>
  </div>
);

export default FeedCard;

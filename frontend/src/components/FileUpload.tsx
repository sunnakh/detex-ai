'use client';

import { useRef, useState, useCallback } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  disabled: boolean;
  selectedFile: File | null;
}

const ALLOWED = ['.txt', '.pdf', '.docx'];
const ALLOWED_MIME = [
  'text/plain',
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
];

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function FileUpload({ onFileSelect, disabled, selectedFile }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File | null) => {
      if (!file) return;
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED.includes(ext)) {
        alert(`Unsupported file type. Please upload ${ALLOWED.join(', ')}.`);
        return;
      }
      onFileSelect(file);
    },
    [onFileSelect],
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0] ?? null);
    // reset so same file can be re-selected
    e.target.value = '';
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [disabled, handleFile],
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  return (
    <>
      {/* Hidden native file input */}
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED.join(',')}
        onChange={handleInputChange}
        style={{ display: 'none' }}
        tabIndex={-1}
      />

      {/* Drop zone overlay — only visible when dragging */}
      {dragOver && (
        <div className="drop-overlay" onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}>
          <span className="drop-overlay-text">📂 Drop your file here</span>
        </div>
      )}

      {selectedFile ? (
        /* File badge — shows staged file */
        <div className="upload-badge">
          <span className="upload-badge-icon">{fileIcon(selectedFile.name)}</span>
          <span className="upload-badge-name">{selectedFile.name}</span>
          <span className="upload-badge-size">{formatBytes(selectedFile.size)}</span>
          <button
            className="upload-badge-remove"
            onClick={() => onFileSelect(null)}
            aria-label="Remove file"
            disabled={disabled}
          >
            ✕
          </button>
        </div>
      ) : (
        /* Upload pill button */
        <button
          className="upload-btn"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          aria-label="Upload a file"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          type="button"
        >
          <PaperclipIcon />
          <span>Upload file</span>
        </button>
      )}
    </>
  );
}

function fileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase();
  if (ext === 'pdf') return '📄';
  if (ext === 'docx') return '📝';
  return '📃';
}

const PaperclipIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.2-9.19a4 4 0 0 1 5.66 5.65L9.64 17.2a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
  </svg>
);

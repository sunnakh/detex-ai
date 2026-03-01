// Shared animated magnifying-glass logo mark for detex.ai
export default function LogoMark({ size = 30 }: { size?: number }) {
  return (
    <span
      style={{
        width: size,
        height: size,
        background: 'var(--accent)',
        borderRadius: Math.round(size * 0.27),
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        boxShadow: '0 2px 8px var(--accent-glow)',
      }}
    >
      <svg
        width={size * 0.58}
        height={size * 0.58}
        viewBox="0 0 20 20"
        fill="none"
        style={{ display: 'block' }}
      >
        <style>{`
          @keyframes detex-scan {
            0%,100% { transform: rotate(-8deg) scale(1); }
            50%      { transform: rotate(8deg) scale(1.08); }
          }
          .detex-lens { animation: detex-scan 2.4s ease-in-out infinite; transform-origin: 8px 8px; }
        `}</style>
        <g className="detex-lens">
          <circle cx="8" cy="8" r="5" stroke="white" strokeWidth="2" fill="none"/>
          <circle cx="8" cy="8" r="2.2" fill="white" fillOpacity="0.25"/>
        </g>
        <line x1="12" y1="12" x2="17" y2="17" stroke="white" strokeWidth="2.2" strokeLinecap="round"/>
      </svg>
    </span>
  );
}

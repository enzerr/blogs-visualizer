import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import rawData from "./public-clusters.json";

// --- Vector Math ---
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

function kMeans(vectors, k, iterations=50) {
  const n = vectors.length;
  const dim = vectors[0].length;
  if (n <= k) return vectors.map((_,i) => i);
  const centroids = [vectors[Math.floor(Math.random()*n)]];
  while (centroids.length < k) {
    const dists = vectors.map(v => {
      let minD = Infinity;
      centroids.forEach(c => { const d = 1-cosineSim(v,c); minD = Math.min(minD, d); });
      return minD * minD;
    });
    const total = dists.reduce((a,b)=>a+b,0);
    let r = Math.random() * total;
    for (let i=0;i<n;i++) { r -= dists[i]; if (r<=0) { centroids.push(vectors[i]); break; } }
    if (centroids.length < k) centroids.push(vectors[Math.floor(Math.random()*n)]);
  }
  let labels = new Array(n).fill(0);
  for (let iter=0; iter<iterations; iter++) {
    const newLabels = vectors.map(v => {
      let best=0, bestSim=-Infinity;
      centroids.forEach((c,ci) => { const s=cosineSim(v,c); if(s>bestSim){bestSim=s;best=ci;} });
      return best;
    });
    for (let ci=0;ci<k;ci++) {
      const members = vectors.filter((_,i)=>newLabels[i]===ci);
      if (members.length===0) continue;
      const newC = new Array(dim).fill(0);
      members.forEach(v => v.forEach((val,d) => { newC[d]+=val/members.length; }));
      const mag = Math.sqrt(newC.reduce((s,v)=>s+v*v,0))+1e-8;
      centroids[ci] = newC.map(v=>v/mag);
    }
    if (newLabels.every((l,i)=>l===labels[i])) break;
    labels = newLabels;
  }
  return labels;
}

function tfidfVectors(docs) {
  const stopwords = new Set(["the","a","an","is","it","in","on","at","to","of","and","or","for","with","this","that","are","was","were","be","have","do","but","not","from","by","as","i","we","you","they"]);
  const tokenize = t => t.toLowerCase().replace(/[^a-z0-9 ]/g," ").split(/\s+/).filter(w=>w.length>2&&!stopwords.has(w));
  const docTokens = docs.map(tokenize);
  const allTerms = {};
  docTokens.forEach(tokens => {
    const seen = new Set();
    tokens.forEach(t => { allTerms[t]=(allTerms[t]||0)+(seen.has(t)?0:1); seen.add(t); });
  });
  const vocab = Object.entries(allTerms).filter(([,df]) => df > 1).sort((a,b)=>b[1]-a[1]).slice(0,500).map(e=>e[0]);
  const idf = {};
  const n = docs.length;
  vocab.forEach(t => { idf[t] = Math.log((n+1)/((allTerms[t]||0)+1))+1; });
  return docTokens.map(tokens => {
    const tf = {};
    tokens.forEach(t => { tf[t]=(tf[t]||0)+1; });
    const vec = vocab.map(t => (tf[t]||0)/tokens.length * idf[t]);
    const mag = Math.sqrt(vec.reduce((s,v)=>s+v*v,0))+1e-8;
    return vec.map(v=>v/mag); 
  });
}

const CLUSTER_COLORS = ["#FF6B6B","#4ECDC4","#FFE66D","#A8E6CF","#FF8B94","#C3A6FF","#FFD93D","#6BCB77","#4D96FF","#FF922B","#F06595","#74C0FC","#A9E34B","#FFA94D","#DA77F2","#20B2AA","#FF7F50","#87CEFA","#DDA0DD","#F0E68C","#E6E6FA","#98FB98"];
const K_VALUES = [1, 2, 3, 5, 8, 13, 21, 34];

export default function Viewer() {
  const [activeK, setActiveK] = useState(5);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [, forceRender] = useState({});
  
  const [viewBox, setViewBox] = useState({ x: -2.5, y: -2.5, w: 5, h: 5 });
  const svgRef = useRef(null);
  const isPanning = useRef(false);
  const panStart = useRef(null);

  const requestRef = useRef();
  const iterRef = useRef(0);
  const posRef = useRef([]);
  const activeKRef = useRef(activeK);

  const { posts, simMatrix, kClusterings } = useMemo(() => {
    const vectors = tfidfVectors(rawData.map(p => p.fullText || `${p.title} ${p.summary}`));
    const n = vectors.length;
    const clean = vectors.map(v => v.map(x => (isFinite(x) ? x : 0)));
    
    const mat = Array.from({length:n}, (_,i) =>
      Array.from({length:n}, (_,j) => {
        if (i===j) return 1;
        const s = cosineSim(clean[i], clean[j]);
        return isFinite(s) ? Math.max(-1, Math.min(1, s)) : 0;
      })
    );
    
    const clusterings = {};
    K_VALUES.forEach(k => {
      const labels = kMeans(vectors, k);
      const clusterLabels = {};
      for (let c=0; c<k; c++) {
        const postsInC = rawData.filter((_, i) => labels[i] === c);
        const kwCounts = {};
        postsInC.forEach(p => {
           (p.keywords || []).forEach(kw => {
              if (kw.length > 2) kwCounts[kw] = (kwCounts[kw]||0) + 1;
           });
        });
        const topKws = Object.entries(kwCounts).sort((a,b)=>b[1]-a[1]).slice(0,2).map(e=>e[0]);
        clusterLabels[c] = k === 1 ? "All Articles" : (topKws.map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" & ") || `Topic ${c+1}`);
      }
      clusterings[k] = { assignments: labels, clusterLabels };
    });

    posRef.current = rawData.map(p => p.pos ? [...p.pos] : [(Math.random()-0.5)*2, (Math.random()-0.5)*2]);
    return { posts: rawData, simMatrix: mat, kClusterings: clusterings };
  }, []);

  useEffect(() => { activeKRef.current = activeK; }, [activeK]);

  const updatePhysics = useCallback(() => {
    iterRef.current += 1;
    const lr = 0.008 * Math.exp(-iterRef.current / 250); 
    
    if (lr > 0.001) {
      const n = posts.length;
      const forces = Array.from({length: n}, () => [0,0]);
      const currentPos = posRef.current;
      const mat = simMatrix;
      const currentAssignments = kClusterings[activeKRef.current].assignments;
      
      // 1. Calculate the center of mass for each cluster (The Magnets)
      const clusterCenters = {};
      const clusterCounts = {};
      for (let i = 0; i < n; i++) {
        const cid = currentAssignments[i];
        if (!clusterCenters[cid]) { clusterCenters[cid] = [0,0]; clusterCounts[cid] = 0; }
        clusterCenters[cid][0] += currentPos[i][0];
        clusterCenters[cid][1] += currentPos[i][1];
        clusterCounts[cid]++;
      }
      for (const cid in clusterCenters) {
        clusterCenters[cid][0] /= clusterCounts[cid];
        clusterCenters[cid][1] /= clusterCounts[cid];
      }
      
      for (let i = 0; i < n; i++) {
        const myCid = currentAssignments[i];
        
        // Gentle global gravity
        forces[i][0] -= currentPos[i][0] * 0.005; 
        forces[i][1] -= currentPos[i][1] * 0.005;

        // NEW: Unbreakable Cluster Magnet (Pulls stragglers home)
        forces[i][0] += (clusterCenters[myCid][0] - currentPos[i][0]) * 0.1;
        forces[i][1] += (clusterCenters[myCid][1] - currentPos[i][1]) * 0.1;

        for (let j = i+1; j < n; j++) {
          const dx = currentPos[i][0] - currentPos[j][0];
          const dy = currentPos[i][1] - currentPos[j][1];
          const d = Math.sqrt(dx*dx + dy*dy) + 1e-8;
          const s = mat[i][j];
          
          const sameCluster = myCid === currentAssignments[j];
          
          const targetD = sameCluster ? Math.max(0.2, (1-s)*0.4) : 1.2 + (1-s)*0.5;
          let f = (d - targetD) * (sameCluster ? 0.3 : 0.1);
          
          if (d < 0.1) f -= (0.1 - d) * 2.0; 
          
          f = Math.max(-1.5, Math.min(1.5, f));
          
          if (!isFinite(f)) continue;
          forces[i][0] -= f * dx/d; forces[i][1] -= f * dy/d;
          forces[j][0] += f * dx/d; forces[j][1] += f * dy/d;
        }
      }
      
      for (let i = 0; i < n; i++) {
        currentPos[i][0] += forces[i][0] * lr;
        currentPos[i][1] += forces[i][1] * lr;
      }
      
      forceRender({}); 
      requestRef.current = requestAnimationFrame(updatePhysics);
    }
  }, [posts.length, simMatrix, kClusterings]);

  useEffect(() => {
    iterRef.current = 0; 
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    requestRef.current = requestAnimationFrame(updatePhysics);
    return () => cancelAnimationFrame(requestRef.current);
  }, [activeK, updatePhysics]);

  const handleWheel = useCallback((e) => {
    const factor = e.deltaY > 0 ? 1.15 : 0.87;
    setViewBox(vb => {
      const svg = svgRef.current;
      if (!svg) return vb;
      const rect = svg.getBoundingClientRect();
      const cursorX = vb.x + (e.clientX - rect.left) / rect.width * vb.w;
      const cursorY = vb.y + (e.clientY - rect.top) / rect.height * vb.h;
      return { x: cursorX - (cursorX - vb.x) * factor, y: cursorY - (cursorY - vb.y) * factor, w: vb.w * factor, h: vb.h * factor };
    });
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const preventScroll = (e) => { e.preventDefault(); handleWheel(e); };
    svg.addEventListener("wheel", preventScroll, { passive: false });
    return () => svg.removeEventListener("wheel", preventScroll);
  }, [handleWheel]);

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    isPanning.current = true;
    panStart.current = { x: e.clientX, y: e.clientY, vb: {...viewBox} };
  };

  const handleMouseMove = useCallback((e) => {
    if (!isPanning.current || !panStart.current) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const dx = (e.clientX - panStart.current.x) / rect.width * panStart.current.vb.w;
    const dy = (e.clientY - panStart.current.y) / rect.height * panStart.current.vb.h;
    setViewBox({ ...panStart.current.vb, x: panStart.current.vb.x - dx, y: panStart.current.vb.y - dy });
  }, []);

  const handleMouseUp = () => { isPanning.current = false; };

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => { window.removeEventListener("mousemove", handleMouseMove); window.removeEventListener("mouseup", handleMouseUp); };
  }, [handleMouseMove]);

  const currentAssignments = kClusterings[activeK].assignments;
  const currentLabels = kClusterings[activeK].clusterLabels;

  const clusterCounts = {};
  const centroids = {};
  
  currentAssignments.forEach((cid, i) => { 
    clusterCounts[cid] = (clusterCounts[cid]||0)+1; 
    if (!centroids[cid]) centroids[cid] = { x: 0, y: 0, count: 0 };
    centroids[cid].x += posRef.current[i][0];
    centroids[cid].y += posRef.current[i][1];
    centroids[cid].count += 1;
  });

  Object.keys(centroids).forEach(cid => {
    centroids[cid].x /= centroids[cid].count;
    centroids[cid].y /= centroids[cid].count;
  });

  return (
    <div style={{ fontFamily:"'DM Mono', monospace", background:"#0a0a0f", height:"100vh", color:"#e8e8f0", display:"flex", flexDirection:"column" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #333; }
        .spotlight-input:focus { border-color: #4ECDC4 !important; outline: none; box-shadow: 0 0 10px rgba(78,205,196,0.2); }
      `}</style>

      <div style={{ padding:"16px 24px", borderBottom:"1px solid #1e1e2e", display:"flex", alignItems:"center", gap:24, background:"#0d0d15" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:8, height:8, borderRadius:"50%", background:"#4ECDC4" }} />
          <span style={{ fontFamily:"'Syne', sans-serif", fontWeight:800, fontSize:18, color:"#fff" }}>AKAJUAN EXPLORER</span>
        </div>
        
        <div style={{ flex:1, maxWidth:400, position:"relative" }}>
          <span style={{ position:"absolute", left:12, top:8, color:"#555" }}>🔍</span>
          <input 
            type="text" 
            placeholder="Spotlight search..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="spotlight-input"
            style={{ width:"100%", background:"#111118", border:"1px solid #2a2a3e", borderRadius:20, padding:"8px 16px 8px 36px", color:"#fff", fontFamily:"inherit", fontSize:12, transition:"all 0.2s" }}
          />
        </div>

        <div style={{ display:"flex", alignItems:"center", gap:8, marginLeft:"auto" }}>
          <span style={{ fontSize:11, color:"#555" }}>Resolution (k):</span>
          <div style={{ display:"flex", background:"#111118", borderRadius:6, border:"1px solid #1e1e2e", overflow:"hidden" }}>
            {K_VALUES.map(k => (
              <button 
                key={k} 
                onClick={() => setActiveK(k)}
                style={{ background: activeK === k ? "#4ECDC4" : "transparent", color: activeK === k ? "#000" : "#888", border:"none", padding:"6px 12px", fontSize:12, fontWeight: activeK === k ? 700 : 400, cursor:"pointer", transition:"all 0.2s" }}
              >
                {k}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ flex:1, display:"flex", overflow:"hidden" }}>
        
        <div style={{ width:240, borderRight:"1px solid #1a1a2e", padding:"16px", overflowY:"auto" }}>
          <div style={{ fontSize:10, color:"#444", letterSpacing:"1px", marginBottom:12 }}>TOPICS AT K={activeK}</div>
          {Object.entries(currentLabels).map(([id, label]) => (
            <div 
              key={id} 
              onMouseEnter={() => setHoveredCluster(Number(id))}
              onMouseLeave={() => setHoveredCluster(null)}
              style={{ display:"flex", alignItems:"center", gap:8, padding:"8px", borderRadius:6, marginBottom:4, cursor:"pointer", background: hoveredCluster === Number(id) ? "rgba(255,255,255,0.05)" : "transparent" }}
            >
              <div style={{ width:8, height:8, borderRadius:2, background: CLUSTER_COLORS[id % CLUSTER_COLORS.length] }} />
              <span style={{ fontSize:11, flex:1, color: hoveredCluster === null || hoveredCluster === Number(id) ? "#fff" : "#555" }}>{label}</span>
              <span style={{ fontSize:10, color:"#444" }}>{clusterCounts[id]}</span>
            </div>
          ))}
          
          <div style={{ marginTop:20, borderTop:"1px solid #1a1a2e", paddingTop:16 }}>
            <div style={{ fontSize:10, color:"#444", letterSpacing:"1px", marginBottom:8 }}>CONTROLS</div>
            <div style={{ fontSize:10, color:"#444", lineHeight:1.8 }}>
              <div>• Scroll to zoom in/out</div>
              <div>• Click and drag to pan</div>
              <div>• Switch 'k' values to animate</div>
            </div>
          </div>
        </div>

        <div style={{ flex:1, position:"relative", overflow:"hidden" }}>
          <svg 
            ref={svgRef} 
            width="100%" height="100%" 
            viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
            style={{ cursor: isPanning.current ? "grabbing" : "grab", userSelect:"none" }}
            onMouseDown={handleMouseDown}
          >
            {posts.map((post, i) => {
              const [cx, cy] = posRef.current[i] || [0,0];
              const clusterId = currentAssignments[i];
              const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
              
              const searchStr = search.toLowerCase();
              const isMatch = !searchStr || 
                (post.title && post.title.toLowerCase().includes(searchStr)) || 
                (post.summary && post.summary.toLowerCase().includes(searchStr)) || 
                (post.fullText && post.fullText.toLowerCase().includes(searchStr)) || 
                (post.keywords && post.keywords.some(kw => kw.toLowerCase().includes(searchStr)));
              
              const isHovered = hoveredCluster !== null && hoveredCluster !== clusterId;
              const isSelected = selected?.url === post.url;
              
              let opacity = 0.8;
              if (searchStr && !isMatch) opacity = 0.05;
              else if (isHovered && !isSelected) opacity = 0.1;
              else if (isSelected) opacity = 1;

              return (
                <circle
                  key={post.url}
                  cx={cx} cy={cy}
                  r={isSelected ? 0.05 : (isMatch && searchStr ? 0.04 : 0.03)}
                  fill={color}
                  opacity={opacity}
                  stroke={isSelected ? "#fff" : "transparent"}
                  strokeWidth={0.01}
                  style={{ transition:"r 0.2s, opacity 0.2s", cursor:"pointer" }}
                  onClick={() => setSelected(post)}
                />
              );
            })}
            
            {Object.keys(centroids).map(cid => {
              if (hoveredCluster !== null && hoveredCluster !== Number(cid)) return null;
              const { x, y } = centroids[cid];
              return (
                <text key={`label-${cid}`} x={x} y={y - 0.25} textAnchor="middle"
                  fontSize="0.1"
                  style={{ 
                    fontFamily:"'Syne',sans-serif", 
                    fontWeight:700, 
                    fill: CLUSTER_COLORS[cid % CLUSTER_COLORS.length], 
                    opacity: 0.9, 
                    pointerEvents: "none",
                    letterSpacing: "0" /* The golden bullet that fixes the exploding text */
                  }}>
                  {currentLabels[cid]}
                </text>
              );
            })}
          </svg>
        </div>

        {selected && (
          <div style={{ width:320, borderLeft:"1px solid #1a1a2e", padding:"24px", overflowY:"auto", background:"#0d0d15" }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:16 }}>
              <div style={{ fontSize:10, color:"#444", letterSpacing:"1px" }}>READING</div>
              <button onClick={() => setSelected(null)} style={{ background:"none", border:"none", color:"#888", cursor:"pointer", fontSize:16 }}>×</button>
            </div>
            
            <div style={{ display:"inline-flex", alignItems:"center", gap:6, marginBottom:16, background:"rgba(255,255,255,0.05)", borderRadius:4, padding:"4px 8px" }}>
              <div style={{ width:6, height:6, borderRadius:1, background: CLUSTER_COLORS[currentAssignments[posts.indexOf(selected)] % CLUSTER_COLORS.length] }} />
              <span style={{ fontSize:10, color:"#ccc" }}>{currentLabels[currentAssignments[posts.indexOf(selected)]]}</span>
            </div>
            
            <div style={{ fontFamily:"'Syne',sans-serif", fontWeight:800, fontSize:18, lineHeight:1.3, marginBottom:16, color:"#fff" }}>
              {selected.title}
            </div>
            
            <p style={{ fontSize:13, color:"#aaa", lineHeight:1.6, marginBottom:20 }}>
              {selected.summary}
            </p>
            
            <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginBottom:24 }}>
              {selected.keywords?.map((kw, i) => (
                <span key={i} style={{ fontSize:10, background:"#1a1a2e", color:"#777", borderRadius:4, padding:"4px 8px" }}>{kw}</span>
              ))}
            </div>
            
            <a href={selected.url} target="_blank" rel="noopener noreferrer"
              style={{ display:"block", background:"#4ECDC4", color:"#000", borderRadius:6, padding:"12px", fontSize:12, fontFamily:"'Syne',sans-serif", fontWeight:800, textDecoration:"none", textAlign:"center" }}>
              READ FULL POST →
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
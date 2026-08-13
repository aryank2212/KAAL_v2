import { MapContainer, TileLayer, Marker } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function MapPanel() {
  return (
    <MapContainer
      center={[28.6, 77.2]}
      zoom={5}
      className="h-[250px] rounded-xl"
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <Marker position={[28.6, 77.2]} />
    </MapContainer>
  );
}

export default MapPanel;
// Author: Sarala Biswal

import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: Number(__ENV.VUS || 5),
  duration: __ENV.DURATION || "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<200"],
  },
};

const baseUrl = __ENV.BASE_URL || "http://127.0.0.1:8000";

export default function () {
  const payload = JSON.stringify({
    k: 5,
    query_features: {
      adult_count: 2,
      child_count: 0,
      length_of_stay: 2,
      booking_window: 14,
      destination_id: "dest-001",
      point_of_sale: "US",
      geo_location_country: "US",
      is_mobile: false,
      sort_type: "recommended",
    },
  });
  const response = http.post(`${baseUrl}/rank`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(response, {
    "rank status is 200": (value) => value.status === 200,
  });
  sleep(1);
}

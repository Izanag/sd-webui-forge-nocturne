(() => {
    "use strict";

    const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
    const rounded = (value) => Math.round(value * 1000000) / 1000000;
    let drag = null;

    function canvasPoint(svg, event) {
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        const transformed = point.matrixTransform(svg.getScreenCTM().inverse());
        return {x: clamp(transformed.x / 1000, 0, 1), y: clamp(transformed.y / 1000, 0, 1)};
    }

    function polygonPoints(polygon) {
        return Array.from(polygon.points).map((point) => ({x: point.x / 1000, y: point.y / 1000}));
    }

    function renderPolygon(polygon, points) {
        polygon.setAttribute("points", points.map((point) => `${point.x * 1000},${point.y * 1000}`).join(" "));
        const svg = polygon.ownerSVGElement;
        for (const handle of svg.querySelectorAll(`[data-region-id="${polygon.dataset.regionId}"][data-point-index]`)) {
            const point = points[Number(handle.dataset.pointIndex)];
            handle.setAttribute("cx", point.x * 1000);
            handle.setAttribute("cy", point.y * 1000);
        }
    }

    function beginDrag(event) {
        const svg = event.target.closest("#regional_canvas svg");
        if (!svg || event.button !== 0) return;

        const target = event.target.closest("[data-region-id]");
        const regionId = target?.dataset.regionId;
        if (!target || !regionId || regionId !== svg.dataset.selectedRegion) return;

        const start = canvasPoint(svg, event);
        const corner = target.dataset.rectCorner;
        const pointIndex = target.dataset.pointIndex;
        const shape = target.classList.contains("nocturne-region-shape")
            ? target
            : svg.querySelector(`.nocturne-region-shape[data-region-id="${regionId}"]`);
        if (!shape) return;

        if (shape.tagName.toLowerCase() === "rect") {
            drag = {
                mode: corner ? "rect-resize" : "rect-move",
                svg,
                shape,
                regionId,
                corner,
                start,
                original: {
                    x: Number(shape.getAttribute("x")) / 1000,
                    y: Number(shape.getAttribute("y")) / 1000,
                    width: Number(shape.getAttribute("width")) / 1000,
                    height: Number(shape.getAttribute("height")) / 1000,
                },
            };
        } else if (shape.tagName.toLowerCase() === "polygon") {
            drag = {
                mode: pointIndex === undefined ? "polygon-move" : "polygon-point",
                svg,
                shape,
                regionId,
                pointIndex: Number(pointIndex),
                start,
                original: polygonPoints(shape),
            };
        } else {
            return;
        }

        svg.setPointerCapture(event.pointerId);
        event.preventDefault();
    }

    function moveRect(point) {
        const {original, shape, corner, start, mode} = drag;
        let {x, y, width, height} = original;
        if (mode === "rect-move") {
            x = clamp(original.x + point.x - start.x, 0, 1 - width);
            y = clamp(original.y + point.y - start.y, 0, 1 - height);
        } else {
            let left = x;
            let top = y;
            let right = x + width;
            let bottom = y + height;
            if (corner.includes("w")) left = clamp(point.x, 0, right - 0.001);
            if (corner.includes("e")) right = clamp(point.x, left + 0.001, 1);
            if (corner.includes("n")) top = clamp(point.y, 0, bottom - 0.001);
            if (corner.includes("s")) bottom = clamp(point.y, top + 0.001, 1);
            x = left;
            y = top;
            width = right - left;
            height = bottom - top;
        }
        shape.setAttribute("x", x * 1000);
        shape.setAttribute("y", y * 1000);
        shape.setAttribute("width", width * 1000);
        shape.setAttribute("height", height * 1000);
        const corners = {
            nw: [x, y],
            ne: [x + width, y],
            sw: [x, y + height],
            se: [x + width, y + height],
        };
        for (const handle of drag.svg.querySelectorAll(`[data-region-id="${drag.regionId}"][data-rect-corner]`)) {
            const position = corners[handle.dataset.rectCorner];
            handle.setAttribute("cx", position[0] * 1000);
            handle.setAttribute("cy", position[1] * 1000);
        }
    }

    function movePolygon(point) {
        let points = drag.original.map((item) => ({...item}));
        if (drag.mode === "polygon-point") {
            points[drag.pointIndex] = point;
        } else {
            const minimumX = Math.min(...points.map((item) => item.x));
            const maximumX = Math.max(...points.map((item) => item.x));
            const minimumY = Math.min(...points.map((item) => item.y));
            const maximumY = Math.max(...points.map((item) => item.y));
            const dx = clamp(point.x - drag.start.x, -minimumX, 1 - maximumX);
            const dy = clamp(point.y - drag.start.y, -minimumY, 1 - maximumY);
            points = points.map((item) => ({x: item.x + dx, y: item.y + dy}));
        }
        renderPolygon(drag.shape, points);
    }

    function moveDrag(event) {
        if (!drag) return;
        const point = canvasPoint(drag.svg, event);
        if (drag.mode.startsWith("rect")) moveRect(point);
        else movePolygon(point);
        event.preventDefault();
    }

    function commitDrag(event) {
        if (!drag) return;
        const current = drag;
        drag = null;
        if (current.svg.hasPointerCapture(event.pointerId)) current.svg.releasePointerCapture(event.pointerId);

        let payload;
        if (current.mode.startsWith("rect")) {
            payload = {
                type: "rect",
                region_id: current.regionId,
                x: rounded(Number(current.shape.getAttribute("x")) / 1000),
                y: rounded(Number(current.shape.getAttribute("y")) / 1000),
                width: rounded(Number(current.shape.getAttribute("width")) / 1000),
                height: rounded(Number(current.shape.getAttribute("height")) / 1000),
            };
        } else {
            payload = {
                type: "polygon",
                region_id: current.regionId,
                points: polygonPoints(current.shape).map((point) => ({x: rounded(point.x), y: rounded(point.y)})),
            };
        }

        const bridgeRoot = gradioApp().querySelector("#regional_geometry_pointer_bridge");
        const bridge = bridgeRoot?.matches("textarea, input") ? bridgeRoot : bridgeRoot?.querySelector("textarea, input");
        if (!bridge) return;
        bridge.value = JSON.stringify(payload);
        if (typeof updateInput === "function") updateInput(bridge);
        else bridge.dispatchEvent(new Event("input", {bubbles: true}));
    }

    onUiLoaded(() => {
        const root = gradioApp();
        root.addEventListener("pointerdown", beginDrag);
        root.addEventListener("pointermove", moveDrag);
        root.addEventListener("pointerup", commitDrag);
        root.addEventListener("pointercancel", () => {
            drag = null;
        });
    });

    window.switch_to_regional = function () {
        const regionalTab = Array.from(gradioApp().querySelectorAll("#tabs button"))
            .find((button) => button.textContent.trim() === "Regional");
        if (regionalTab) regionalTab.click();
        return Array.from(arguments);
    };
})();

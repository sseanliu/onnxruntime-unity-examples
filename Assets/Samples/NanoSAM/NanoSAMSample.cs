using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[RequireComponent(typeof(VirtualTextureSource), typeof(NanoSAMVisualizer))]
public sealed class NanoSAMSample : MonoBehaviour
{
    [Header("NanoSAM Options")]
    [SerializeField]
    private RemoteFile encoderModelFile = new("https://huggingface.co/asus4/nanosam-ort/resolve/main/resnet18_image_encoder.with_runtime_opt.ort?download=true");

    [SerializeField]
    private RemoteFile decoderModelFile = new("https://huggingface.co/asus4/nanosam-ort/resolve/main/mobile_sam_mask_decoder.with_runtime_opt.ort?download=true");

    [SerializeField]
    private NanoSAM.Options options;

    [Header("UI")]
    [SerializeField]
    private RectTransform preview;

    [SerializeField]
    private GameObject loadingIndicator;

    [SerializeField]
    private Button resetButton;

    [SerializeField]
    private TMPro.TMP_Dropdown maskDropdown;

    [SerializeField]
    private Image positivePointPrefab;

    private readonly List<NanoSAM.Point> points = new();
    private NanoSAM inference;
    private Texture inputTexture;
    private NanoSAMVisualizer visualizer;

    private Vector2? trackingPoint = null;
    private Vector2? smoothedTrackingPoint = null; // Smoothed tracking point
    private Image trackingPointImage;

    private async void Start()
    {
        // Show loading indicator
        loadingIndicator.SetActive(true);

        // Load model files
        var token = destroyCancellationToken;
        byte[] encoderModel = await encoderModelFile.Load(token);
        byte[] decoderModel = await decoderModelFile.Load(token);

        inference = new NanoSAM(encoderModel, decoderModel, options);
        visualizer = GetComponent<NanoSAMVisualizer>();

        // Register click event
        var callback = new EventTrigger.TriggerEvent();
        callback.AddListener((data) => OnPointerDown((PointerEventData)data));
        var trigger = preview.gameObject.AddComponent<EventTrigger>();
        trigger.triggers.Add(new()
        {
            eventID = EventTriggerType.PointerDown,
            callback = callback,
        });

        // Listen for texture updates
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTexture);
        }

        // Setup UI
        maskDropdown.ClearOptions();
        maskDropdown.AddOptions(new List<string> { "Negative", "Positive" });
        maskDropdown.value = 1;

        resetButton.onClick.AddListener(ResetMask);

        // Hide loading indicator
        loadingIndicator.SetActive(false);
    }

    private void Update()
    {
        if (trackingPoint.HasValue && inputTexture != null)
        {
            Run(points);
        }
    }

    private void OnDestroy()
    {
        resetButton?.onClick.RemoveListener(ResetMask);

        if (preview != null && preview.TryGetComponent(out EventTrigger trigger))
        {
            Destroy(trigger);
        }

        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTexture);
        }

        inference?.Dispose();
    }

    private void OnTexture(Texture texture)
    {
        inputTexture = texture;
    }

    private void OnPointerDown(PointerEventData data)
    {
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(preview, data.position, data.pressEventCamera, out Vector2 rectPosition))
        {
            return;
        }

        // Normalize coordinates and flip Y axis
        Vector2 point = Rect.PointToNormalized(preview.rect, rectPosition);
        point.y = 1.0f - point.y;

        trackingPoint = point;

        points.Clear();
        points.Add(new NanoSAM.Point(trackingPoint.Value, 1));

        if (trackingPointImage == null)
        {
            trackingPointImage = Instantiate(positivePointPrefab, preview);
        }
        trackingPointImage.rectTransform.anchoredPosition = rectPosition;

        Debug.Log($"Tracking initialized at: {trackingPoint}");
        Run(points);
    }

    private void ResetMask()
    {
        trackingPoint = null;
        smoothedTrackingPoint = null;
        points.Clear();

        if (trackingPointImage != null)
        {
            Destroy(trackingPointImage.gameObject);
            trackingPointImage = null;
        }

        inference.ResetOutput();
        visualizer.UpdateMask(inference.OutputMask, inputTexture);
    }

    private void Run(List<NanoSAM.Point> points)
    {
        if (inputTexture == null) return;

        inference.Run(inputTexture, points.AsReadOnly());
        visualizer.UpdateMask(inference.OutputMask, inputTexture);

        UpdateTrackingPoint(inference.OutputMask.ToArray(), 256, 256); // Adjust to dynamic resolution
    }

    private void UpdateTrackingPoint(float[] mask, int width, int height)
    {
        float sumX = 0, sumY = 0, totalWeight = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float value = mask[y * width + x];

                if (value > 0.5f) // Threshold
                {
                    sumX += x * value;
                    sumY += y * value;
                    totalWeight += value;
                }
            }
        }

        if (totalWeight > 0)
        {
            Vector2 newPoint = new(sumX / totalWeight / width, 1.0f - (sumY / totalWeight / height));

            // Smoothed tracking point
            smoothedTrackingPoint = smoothedTrackingPoint.HasValue
                ? Vector2.Lerp(smoothedTrackingPoint.Value, newPoint, 0.3f) // Linear interpolation
                : newPoint;

            trackingPoint = smoothedTrackingPoint;

            points.Clear();
            points.Add(new NanoSAM.Point(trackingPoint.Value, 1));

            if (trackingPointImage != null)
            {
                Vector2 rectPosition = Rect.NormalizedToPoint(preview.rect, new Vector2(trackingPoint.Value.x, 1.0f - trackingPoint.Value.y));
                trackingPointImage.rectTransform.anchoredPosition = rectPosition;
            }
        }
        else
        {
            trackingPoint = null;
        }
    }
}

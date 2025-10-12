"""Signal handler system for notifications and alerts"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import time
from datetime import datetime, timedelta

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

try:
    from win10toast import ToastNotifier
    WIN10TOAST_AVAILABLE = True
except ImportError:
    WIN10TOAST_AVAILABLE = False

from .config import Config


class SignalHandler(ABC):
    """Abstract base class for signal handlers"""
    
    @abstractmethod
    def trigger(self, event: Dict) -> bool:
        """Trigger the signal with given event data
        
        Args:
            event: Dictionary containing event information
        
        Returns:
            True if signal was successfully triggered, False otherwise
        """
        pass


class DesktopNotificationHandler(SignalHandler):
    """Desktop notification implementation"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.last_notification_time = None
        self.cooldown = timedelta(seconds=config.notification.cooldown_seconds)
        
        # Initialize notifier based on available libraries
        self.notifier = None
        self.notification_method = None
        
        if WIN10TOAST_AVAILABLE:
            self.notifier = ToastNotifier()
            self.notification_method = "win10toast"
        elif PLYER_AVAILABLE:
            self.notification_method = "plyer"
        else:
            self.notification_method = "console"
            print("Warning: No notification library available. Using console output.")
    
    def trigger(self, event: Dict) -> bool:
        """Trigger desktop notification
        
        Args:
            event: Dictionary containing:
                - event_type: 'distracted' or 'focused'
                - timestamp: datetime of event
                - score_data: scoring information
                - prediction_data: prediction information
        
        Returns:
            True if notification was sent, False if on cooldown or failed
        """
        if not self.config.notification.enabled:
            return False
        
        # Check cooldown
        now = datetime.now()
        if self.last_notification_time is not None:
            if now - self.last_notification_time < self.cooldown:
                return False
        
        # Prepare message
        title = self.config.notification.title
        message = self.config.notification.distracted_message
        
        if event.get('event_type') == 'distracted':
            score = event.get('score_data', {}).get('lock_in_score', 0)
            message = f"{message}\nLock-in score: {score:.2f}"
        
        # Send notification
        success = False
        try:
            if self.notification_method == "win10toast":
                self.notifier.show_toast(
                    title,
                    message,
                    duration=5,
                    threaded=True
                )
                success = True
            elif self.notification_method == "plyer":
                notification.notify(
                    title=title,
                    message=message,
                    timeout=5
                )
                success = True
            else:
                # Console fallback
                print(f"\n{'='*50}")
                print(f"NOTIFICATION: {title}")
                print(f"{message}")
                print(f"{'='*50}\n")
                success = True
        except Exception as e:
            print(f"Failed to send notification: {e}")
            success = False
        
        if success:
            self.last_notification_time = now
        
        return success


class LoggingSignalHandler(SignalHandler):
    """Signal handler that logs events"""
    
    def __init__(self, log_file: str = "events.log"):
        self.log_file = log_file
    
    def trigger(self, event: Dict) -> bool:
        """Log event to file"""
        try:
            with open(self.log_file, 'a') as f:
                timestamp = event.get('timestamp', datetime.now())
                event_type = event.get('event_type', 'unknown')
                score = event.get('score_data', {}).get('lock_in_score', 0)
                f.write(f"{timestamp} | {event_type} | score={score:.3f}\n")
            return True
        except Exception as e:
            print(f"Failed to log event: {e}")
            return False


class CompositeSignalHandler(SignalHandler):
    """Composite handler that triggers multiple handlers"""
    
    def __init__(self):
        self.handlers = []
    
    def add_handler(self, handler: SignalHandler):
        """Add a handler to the composite"""
        self.handlers.append(handler)
    
    def trigger(self, event: Dict) -> bool:
        """Trigger all handlers"""
        results = []
        for handler in self.handlers:
            try:
                result = handler.trigger(event)
                results.append(result)
            except Exception as e:
                print(f"Handler {handler.__class__.__name__} failed: {e}")
                results.append(False)
        
        return any(results)


class HardwareSignalHandler(SignalHandler):
    """
    Extensible handler for future hardware integration
    (e.g., Bluetooth shock device, smart lights, etc.)
    """
    
    def __init__(self, device_config: Optional[Dict] = None):
        self.device_config = device_config or {}
        # Future: Initialize hardware connection here
    
    def trigger(self, event: Dict) -> bool:
        """
        Trigger hardware signal
        
        This is a placeholder for future implementation.
        Subclass this and implement device-specific logic.
        """
        print("Hardware signal triggered (placeholder implementation)")
        # Future: Send signal to hardware device
        return True


def create_signal_handler(config: Config) -> SignalHandler:
    """Factory function to create appropriate signal handler
    
    Args:
        config: Configuration object
    
    Returns:
        Configured signal handler (may be composite)
    """
    composite = CompositeSignalHandler()
    
    # Add desktop notification handler
    if config.notification.enabled:
        composite.add_handler(DesktopNotificationHandler(config))
    
    # Could add more handlers here based on config
    # e.g., if config has hardware settings, add HardwareSignalHandler
    
    return composite


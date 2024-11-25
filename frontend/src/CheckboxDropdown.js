import React from "react";
import { Button, ButtonGroup, Dropdown, Form } from "react-bootstrap";

const CheckboxMenu = React.forwardRef(
  (
    {
      children,
      style,
      className,
      "aria-labelledby": labeledBy,
      onSelectAll,
      onSelectNone,
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        style={style}
        className={`${className} CheckboxMenu`}
        aria-labelledby={labeledBy}
      >
        <div
          className="d-flex flex-column"
          style={{ maxHeight: "calc(100vh)", overflow: "none" }}
        >
          <ul
            className="list-unstyled flex-shrink mb-0"
            style={{ overflow: "auto" }}
          >
            {children}
          </ul>
          <div className="dropdown-item border-top pt-2 pb-0">
            <ButtonGroup size="sm">
              <Button variant="link" onClick={onSelectAll}>
                Select All
              </Button>
              <Button variant="link" onClick={onSelectNone}>
                Select None
              </Button>
            </ButtonGroup>
          </div>
        </div>
      </div>
    );
  }
);

const CheckDropdownItem = React.forwardRef(
  ({ children, id, checked, onChange }, ref) => {
    return (
      <Form.Group ref={ref} className="dropdown-item mb-0" controlId={id}>
        <Form.Check
          type="checkbox"
          label={children}
          checked={checked}
          onChange={(e) => onChange(id, e.target.checked)} // Use passed handler
        />
      </Form.Group>
    );
  }
);

export const CheckboxDropdown = ({ items, setItems }) => {
  const handleChecked = (key, isChecked) => {
    // Update state using `setItems`
    setItems((prevItems) =>
      prevItems.map((item) =>
        item.id === key ? { ...item, checked: isChecked } : item
      )
    );
  };

  const handleSelectAll = () => {
    // Set all items to checked
    setItems((prevItems) =>
      prevItems.map((item) => ({ ...item, checked: true }))
    );
  };

  const handleSelectNone = () => {
    // Set all items to unchecked
    setItems((prevItems) =>
      prevItems.map((item) => ({ ...item, checked: false }))
    );
  };

  return (
    <Dropdown>
      <Dropdown.Toggle variant="primary" id="dropdown-basic">
        Properties
      </Dropdown.Toggle>

      <Dropdown.Menu
        as={CheckboxMenu}
        onSelectAll={handleSelectAll}
        onSelectNone={handleSelectNone}
      >
        {items.map((item) => (
          <Dropdown.Item
            key={item.id}
            as={CheckDropdownItem}
            id={item.id}
            checked={item.checked}
            onChange={handleChecked} // Pass handler
          >
            {item.label}
          </Dropdown.Item>
        ))}
      </Dropdown.Menu>
    </Dropdown>
  );
};
